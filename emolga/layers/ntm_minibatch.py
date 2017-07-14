__author__ = 'jiataogu'
import theano
import theano.tensor as T

import scipy.linalg as sl
import numpy as np
from .core import *
from .recurrent import *
import copy

"""
This implementation supports both minibatch learning and on-line training.
We need a minibatch version for Neural Turing Machines.
"""


class Reader(Layer):
    """
        "Reader Head" of the Neural Turing Machine.
    """

    def __init__(self, input_dim, memory_width, shift_width, shift_conv,
                 init='glorot_uniform', inner_init='orthogonal',
                 name=None):
        super(Reader, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_width

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)

        self.tanh = activations.get('tanh')
        self.sigmoid = activations.get('sigmoid')
        self.softplus = activations.get('softplus')
        self.vec_softmax = activations.get('vector_softmax')
        self.softmax = activations.get('softmax')

        """
        Reader Params.
        """
        self.W_key = self.init((input_dim, memory_width))
        self.W_shift = self.init((input_dim, shift_width))
        self.W_beta = self.init(input_dim)
        self.W_gama = self.init(input_dim)
        self.W_g = self.init(input_dim)

        self.b_key = shared_zeros(memory_width)
        self.b_shift = shared_zeros(shift_width)
        self.b_beta = theano.shared(floatX(0))
        self.b_gama = theano.shared(floatX(0))
        self.b_g = theano.shared(floatX(0))

        self.shift_conv = shift_conv

        # add params and set names.
        self.params = [self.W_key, self.W_shift, self.W_beta, self.W_gama, self.W_g,
                       self.b_key, self.b_shift, self.b_beta, self.b_gama, self.b_g]

        self.W_key.name, self.W_shift.name, self.W_beta.name, \
        self.W_gama.name, self.W_g.name = 'W_key', 'W_shift', 'W_beta', \
                                          'W_gama', 'W_g'

        self.b_key.name, self.b_shift.name, self.b_beta.name, \
        self.b_gama.name, self.b_g.name = 'b_key', 'b_shift', 'b_beta', \
                                          'b_gama', 'b_g'

    def __call__(self, X, w_temp, m_temp):
        # input dimensions
        # X:      (nb_samples, input_dim)
        # w_temp: (nb_samples, memory_dim)
        # m_temp: (nb_samples, memory_dim, memory_width) ::tensor_memory

        key = dot(X, self.W_key, self.b_key)  # (nb_samples, memory_width)
        shift = self.softmax(
            dot(X, self.W_shift, self.b_shift))  # (nb_samples, shift_width)

        beta = self.softplus(dot(X, self.W_beta, self.b_beta))[:, None]  # (nb_samples, x)
        gamma = self.softplus(dot(X, self.W_gama, self.b_gama)) + 1.  # (nb_samples,)
        gamma = gamma[:, None]  # (nb_samples, x)
        g = self.sigmoid(dot(X, self.W_g, self.b_g))[:, None]  # (nb_samples, x)

        signal = [key, shift, beta, gamma, g]

        w_c = self.softmax(
            beta * cosine_sim2d(key, m_temp))  # (nb_samples, memory_dim) //content-based addressing
        w_g = g * w_c + (1 - g) * w_temp  # (nb_samples, memory_dim) //history interpolation
        w_s = shift_convolve2d(w_g, shift, self.shift_conv)  # (nb_samples, memory_dim) //convolutional shift
        w_p = w_s ** gamma  # (nb_samples, memory_dim) //sharpening
        w_t = w_p / T.sum(w_p, axis=1)[:, None]  # (nb_samples, memory_dim)
        return w_t


class Writer(Reader):
    """
        "Writer head" of the Neural Turing Machine
    """

    def __init__(self, input_dim, memory_width, shift_width, shift_conv,
                 init='glorot_uniform', inner_init='orthogonal',
                 name=None):
        super(Writer, self).__init__(input_dim, memory_width, shift_width, shift_conv,
                                     init, inner_init, name)

        """
        Writer Params.
        """
        self.W_erase = self.init((input_dim, memory_width))
        self.W_add = self.init((input_dim, memory_width))

        self.b_erase = shared_zeros(memory_width)
        self.b_add = shared_zeros(memory_width)

        # add params and set names.
        self.params += [self.W_erase, self.W_add, self.b_erase, self.b_add]

        self.W_erase.name, self.W_add.name = 'W_erase', 'W_add'
        self.b_erase.name, self.b_add.name = 'b_erase', 'b_add'

    def get_fixer(self, X):
        erase = self.sigmoid(dot(X, self.W_erase, self.b_erase))  # (nb_samples, memory_width)
        add   = self.sigmoid(dot(X, self.W_add, self.b_add))  # (nb_samples, memory_width)
        return erase, add


class Controller(Recurrent):
    """
    Controller used in Neural Turing Machine.
        - Core cell (Memory)
        - Reader head
        - Writer head
    It is a simple RNN version. In reality the Neural Turing Machine will use the LSTM cell.
    """

    def __init__(self,
                 input_dim,
                 memory_dim,
                 memory_width,
                 hidden_dim,
                 shift_width=3,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 name=None,
                 readonly=False,
                 curr_input=False,
                 recurrence=False,
                 memorybook=None
                 ):
        super(Controller, self).__init__()
        # Initialization of the dimensions.
        self.input_dim     = input_dim
        self.memory_dim    = memory_dim
        self.memory_width  = memory_width
        self.hidden_dim    = hidden_dim
        self.shift_width   = shift_width

        self.init          = initializations.get(init)
        self.inner_init    = initializations.get(inner_init)
        self.tanh          = activations.get('tanh')
        self.softmax       = activations.get('softmax')
        self.vec_softmax   = activations.get('vector_softmax')

        self.readonly      = readonly
        self.curr_input    = curr_input
        self.recurrence    = recurrence
        self.memorybook    = memorybook

        """
        Controller Module.
        """
        # hidden projection:
        self.W_in          = self.init((input_dim, hidden_dim))
        self.b_in          = shared_zeros(hidden_dim)
        self.W_rd          = self.init((memory_width, hidden_dim))
        self.W_in.name     = 'W_in'
        self.b_in.name     = 'b_in'
        self.W_rd.name     = 'W_rd'
        self.params        = [self.W_in, self.b_in, self.W_rd]

        # use recurrence:
        if self.recurrence:
            self.W_hh      = self.inner_init((hidden_dim, hidden_dim))
            self.W_hh.name = 'W_hh'
            self.params   += [self.W_hh]

        # Shift convolution
        shift_conv         = sl.circulant(np.arange(memory_dim)).T[
                                np.arange(-(shift_width // 2), (shift_width // 2) + 1)][::-1]

        # use the current input for weights.
        if self.curr_input:
            controller_size = self.input_dim + self.hidden_dim
        else:
            controller_size = self.hidden_dim

        # write head
        if not readonly:
            self.writer    = Writer(controller_size, memory_width, shift_width, shift_conv, name='writer')
            self.writer.set_name('writer')
            self._add(self.writer)

        # read head
        self.reader        = Reader(controller_size, memory_width, shift_width, shift_conv, name='reader')
        self.reader.set_name('reader')
        self._add(self.reader)

        # ***********************************************************
        # reserved for None initialization (we don't use these often)
        self.memory_init   = self.init((memory_dim, memory_width))
        self.w_write_init  = self.softmax(np.random.rand(1, memory_dim).astype(theano.config.floatX))
        self.w_read_init   = self.softmax(np.random.rand(1, memory_dim).astype(theano.config.floatX))
        self.contr_init    = self.tanh(np.random.rand(1, hidden_dim).astype(theano.config.floatX))

        if name is not None:
            self.set_name(name)

    def _controller(self, input_t, read_t, controller_tm1=None):
        # input_t : (nb_sample, input_dim)
        # read_t  : (nb_sample, memory_width)
        # controller_tm1: (nb_sample, hidden_dim)
        if self.recurrence:
            return self.tanh(dot(input_t, self.W_in) +
                             dot(controller_tm1, self.W_hh) +
                             dot(read_t, self.W_rd)  +
                             self.b_in)
        else:
            return self.tanh(dot(input_t, self.W_in) +
                             dot(read_t, self.W_rd)  +
                             self.b_in)

    @staticmethod
    def _read(w_read, memory):
        # w_read : (nb_sample, memory_dim)
        # memory : (nb_sample, memory_dim, memory_width)
        # return dot(w_read, memory)

        return T.sum(w_read[:, :, None] * memory, axis=1)

    @staticmethod
    def _write(w_write, memory, erase, add):
        # w_write: (nb_sample, memory_dim)
        # memory : (nb_sample, memory_dim, memory_width)
        # erase/add: (nb_sample, memory_width)

        w_write  = w_write[:, :, None]
        erase    = erase[:, None, :]
        add      = add[:, None, :]

        m_erased = memory * (1 - w_write * erase)
        memory_t = m_erased + w_write * add  # (nb_sample, memory_dim, memory_width)
        return memory_t

    def _step(self, input_t, mask_t,
              memory_tm1,
              w_write_tm1, w_read_tm1,
              controller_tm1):
        # input_t:     (nb_sample, input_dim)
        # memory_tm1:  (nb_sample, memory_dim, memory_width)
        # w_write_tm1: (nb_sample, memory_dim)
        # w_read_tm1:  (nb_sample, memory_dim)
        # controller_tm1: (nb_sample, hidden_dim)

        # read the memory
        if self.curr_input:
            info     = T.concatenate((controller_tm1, input_t), axis=1)
            w_read_t = self.reader(info, w_read_tm1, memory_tm1)
            read_tm1 = self._read(w_read_t, memory_tm1)
        else:
            read_tm1 = self._read(w_read_tm1, memory_tm1)       # (nb_sample, memory_width)

        # get the new controller (hidden states.)
        if self.recurrence:
            controller_t = self._controller(input_t, read_tm1, controller_tm1)
        else:
            controller_t = self._controller(input_t, read_tm1)  # (nb_sample, controller_size)

        # update the memory cell (if need)
        if not self.readonly:
            if self.curr_input:
                infow          = T.concatenate((controller_t, input_t), axis=1)
                w_write_t      = self.writer(infow, w_write_tm1, memory_tm1)     # (nb_sample, memory_dim)
                erase_t, add_t = self.writer.get_fixer(infow)                    # (nb_sample, memory_width)
            else:
                w_write_t      = self.writer(controller_t, w_write_tm1, memory_tm1)
                erase_t, add_t = self.writer.get_fixer(controller_t)
            memory_t           = self._write(w_write_t, memory_tm1, erase_t, add_t)  # (nb_sample, memory_dim, memory_width)
        else:
            w_write_t          = w_write_tm1
            memory_t           = memory_tm1

        # get the next reading weights.
        if not self.curr_input:
            w_read_t           = self.reader(controller_t, w_read_tm1, memory_t)  # (nb_sample, memory_dim)

        # over masking
        memory_t     = memory_t     * mask_t[:, :, None] + memory_tm1 * (1 - mask_t[:, :, None])
        w_read_t     = w_read_t     * mask_t + w_read_tm1     * (1 - mask_t)
        w_write_t    = w_write_t    * mask_t + w_write_tm1    * (1 - mask_t)
        controller_t = controller_t * mask_t + controller_tm1 * (1 - mask_t)

        return memory_t, w_write_t, w_read_t, controller_t

    def __call__(self, X, mask=None, M=None, init_ww=None,
                 init_wr=None, init_c=None, return_sequence=False,
                 one_step=False, return_full=False):
        # recurrent cell only work for tensor.
        if X.ndim == 2:
            X = X[:, None, :]
        nb_samples = X.shape[0]

        # mask
        if mask is None:
            mask = T.alloc(1., X.shape[0], 1)

        padded_mask = self.get_padded_shuffled_mask(mask, pad=0)
        X = X.dimshuffle((1, 0, 2))

        # ***********************************************************************
        # initialization states
        if M is None:
            memory_init  = T.repeat(self.memory_init[None, :, :], nb_samples, axis=0)
        else:
            memory_init  = M

        if init_wr is None:
            w_read_init  = T.repeat(self.w_read_init, nb_samples, axis=0)
        else:
            w_read_init  = init_wr

        if init_ww is None:
            w_write_init = T.repeat(self.w_write_init, nb_samples, axis=0)
        else:
            w_write_init = init_ww

        if init_c is None:
            contr_init   = T.repeat(self.contr_init, nb_samples, axis=0)
        else:
            contr_init   = init_c
        # ************************************************************************

        outputs_info = [memory_init, w_write_init, w_read_init, contr_init]

        if one_step:
            seq = [X[0], padded_mask[0]]
            outputs = self._step(*(seq + outputs_info))
            return outputs
        else:
            seq = [X, padded_mask]
            outputs, _ = theano.scan(
                self._step,
                sequences=seq,
                outputs_info=outputs_info,
                name='controller_recurrence'
            )

        self.monitor['memory_info']   = outputs[0]
        self.monitor['write_weights'] = outputs[1]
        self.monitor['read_weights']  = outputs[2]

        if not return_full:
            if return_sequence:
                return outputs[-1].dimshuffle((1, 0, 2))
            return outputs[-1][-1]
        else:
            if return_sequence:
                return [a.dimshuffle((1, 0, 2)) for a in outputs]
            return [a[-1] for a in outputs]


class AttentionReader(Layer):
    """
        "Reader Head" of the Neural Turing Machine.
    """

    def __init__(self, input_dim, memory_width, shift_width, shift_conv,
                 init='glorot_uniform', inner_init='orthogonal',
                 name=None):
        super(AttentionReader, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_width

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)

        self.tanh = activations.get('tanh')
        self.sigmoid = activations.get('sigmoid')
        self.softplus = activations.get('softplus')
        self.vec_softmax = activations.get('vector_softmax')
        self.softmax = activations.get('softmax')

        """
        Reader Params.
        """
        self.W_key   = self.init((input_dim, memory_width))
        self.W_lock  = self.inner_init((memory_width, memory_width))

        self.W_shift = self.init((input_dim, shift_width))
        self.W_beta = self.init(input_dim)
        self.W_gama = self.init(input_dim)
        self.W_g = self.init(input_dim)

        # self.v     = self.init(memory_width)
        self.b_key = shared_zeros(memory_width)
        self.b_shift = shared_zeros(shift_width)
        self.b_beta = theano.shared(floatX(0))
        self.b_gama = theano.shared(floatX(0))
        self.b_g = theano.shared(floatX(0))

        self.shift_conv = shift_conv

        # add params and set names.
        self.params = [self.W_key, self.W_shift, self.W_beta, self.W_gama, self.W_g,
                       self.b_key, self.b_shift, self.b_beta, self.b_gama, self.b_g,
                       self.W_lock]

        self.W_key.name, self.W_shift.name, self.W_beta.name, \
        self.W_gama.name, self.W_g.name = 'W_key', 'W_shift', 'W_beta', \
                                          'W_gama', 'W_g'
        self.W_lock.name  = 'W_lock'

        self.b_key.name, self.b_shift.name, self.b_beta.name, \
        self.b_gama.name, self.b_g.name = 'b_key', 'b_shift', 'b_beta', \
                                          'b_gama', 'b_g'

    def __call__(self, X, w_temp, m_temp):
        # input dimensions
        # X:      (nb_samples, input_dim)
        # w_temp: (nb_samples, memory_dim)
        # m_temp: (nb_samples, memory_dim, memory_width) ::tensor_memory

        key   = dot(X, self.W_key, self.b_key)  # (nb_samples, memory_width)
        lock  = dot(m_temp, self.W_lock)        # (nb_samples, memory_dim, memory_width)
        shift = self.softmax(
            dot(X, self.W_shift, self.b_shift))  # (nb_samples, shift_width)

        beta = self.softplus(dot(X, self.W_beta, self.b_beta))[:, None]  # (nb_samples, x)
        gamma = self.softplus(dot(X, self.W_gama, self.b_gama)) + 1.  # (nb_samples,)
        gamma = gamma[:, None]  # (nb_samples, x)
        g = self.sigmoid(dot(X, self.W_g, self.b_g))[:, None]  # (nb_samples, x)

        signal = [key, shift, beta, gamma, g]

        energy = T.sum(key[:, None, :] * lock, axis=2)
        # energy = T.tensordot(key[:, None, :] + lock, self.v, [2, 0])
        w_c    = self.softmax(beta * energy)
        # w_c = self.softmax(
        #     beta * cosine_sim2d(key, m_temp))  # (nb_samples, memory_dim) //content-based addressing
        w_g = g * w_c + (1 - g) * w_temp  # (nb_samples, memory_dim) //history interpolation
        w_s = shift_convolve2d(w_g, shift, self.shift_conv)  # (nb_samples, memory_dim) //convolutional shift
        w_p = w_s ** gamma  # (nb_samples, memory_dim) //sharpening
        w_t = w_p / T.sum(w_p, axis=1)[:, None]  # (nb_samples, memory_dim)
        return w_t


class AttentionWriter(AttentionReader):
    """
        "Writer head" of the Neural Turing Machine
    """

    def __init__(self, input_dim, memory_width, shift_width, shift_conv,
                 init='glorot_uniform', inner_init='orthogonal',
                 name=None):
        super(AttentionWriter, self).__init__(input_dim, memory_width, shift_width, shift_conv,
                                     init, inner_init, name)

        """
        Writer Params.
        """
        self.W_erase = self.init((input_dim, memory_width))
        self.W_add = self.init((input_dim, memory_width))

        self.b_erase = shared_zeros(memory_width)
        self.b_add = shared_zeros(memory_width)

        # add params and set names.
        self.params += [self.W_erase, self.W_add, self.b_erase, self.b_add]

        self.W_erase.name, self.W_add.name = 'W_erase', 'W_add'
        self.b_erase.name, self.b_add.name = 'b_erase', 'b_add'

    def get_fixer(self, X):
        erase = self.sigmoid(dot(X, self.W_erase, self.b_erase))  # (nb_samples, memory_width)
        add   = self.sigmoid(dot(X, self.W_add, self.b_add))  # (nb_samples, memory_width)
        return erase, add



class BernoulliController(Recurrent):
    """
    Controller used in Neural Turing Machine.
        - Core cell (Memory): binary memory
        - Reader head
        - Writer head
    It is a simple RNN version. In reality the Neural Turing Machine will use the LSTM cell.
    """

    def __init__(self,
                 input_dim,
                 memory_dim,
                 memory_width,
                 hidden_dim,
                 shift_width=3,
                 init='glorot_uniform',
                 inner_init='orthogonal',
                 name=None,
                 readonly=False,
                 curr_input=False,
                 recurrence=False,
                 memorybook=None
                 ):
        super(BernoulliController, self).__init__()
        # Initialization of the dimensions.
        self.input_dim     = input_dim
        self.memory_dim    = memory_dim
        self.memory_width  = memory_width
        self.hidden_dim    = hidden_dim
        self.shift_width   = shift_width

        self.init          = initializations.get(init)
        self.inner_init    = initializations.get(inner_init)
        self.tanh          = activations.get('tanh')
        self.softmax       = activations.get('softmax')
        self.vec_softmax   = activations.get('vector_softmax')
        self.sigmoid       = activations.get('sigmoid')

        self.readonly      = readonly
        self.curr_input    = curr_input
        self.recurrence    = recurrence
        self.memorybook    = memorybook

        """
        Controller Module.
        """
        # hidden projection:
        self.W_in          = self.init((input_dim, hidden_dim))
        self.b_in          = shared_zeros(hidden_dim)
        self.W_rd          = self.init((memory_width, hidden_dim))
        self.W_in.name     = 'W_in'
        self.b_in.name     = 'b_in'
        self.W_rd.name     = 'W_rd'
        self.params        = [self.W_in, self.b_in, self.W_rd]

        # use recurrence:
        if self.recurrence:
            self.W_hh      = self.inner_init((hidden_dim, hidden_dim))
            self.W_hh.name = 'W_hh'
            self.params   += [self.W_hh]

        # Shift convolution
        shift_conv         = sl.circulant(np.arange(memory_dim)).T[
                                np.arange(-(shift_width // 2), (shift_width // 2) + 1)][::-1]

        # use the current input for weights.
        if self.curr_input:
            controller_size = self.input_dim + self.hidden_dim
        else:
            controller_size = self.hidden_dim

        # write head
        if not readonly:
            self.writer    = AttentionWriter(controller_size, memory_width, shift_width, shift_conv, name='writer')
            self.writer.set_name('writer')
            self._add(self.writer)

        # read head
        self.reader        = AttentionReader(controller_size, memory_width, shift_width, shift_conv, name='reader')
        self.reader.set_name('reader')
        self._add(self.reader)

        # ***********************************************************
        # reserved for None initialization (we don't use these often)
        self.memory_init   = self.sigmoid(self.init((memory_dim, memory_width)))
        self.w_write_init  = self.softmax(np.random.rand(1, memory_dim).astype(theano.config.floatX))
        self.w_read_init   = self.softmax(np.random.rand(1, memory_dim).astype(theano.config.floatX))
        self.contr_init    = self.tanh(np.random.rand(1, hidden_dim).astype(theano.config.floatX))

        if name is not None:
            self.set_name(name)

    def _controller(self, input_t, read_t, controller_tm1=None):
        # input_t : (nb_sample, input_dim)
        # read_t  : (nb_sample, memory_width)
        # controller_tm1: (nb_sample, hidden_dim)
        if self.recurrence:
            return self.tanh(dot(input_t, self.W_in) +
                             dot(controller_tm1, self.W_hh) +
                             dot(read_t, self.W_rd)  +
                             self.b_in)
        else:
            return self.tanh(dot(input_t, self.W_in) +
                             dot(read_t, self.W_rd)  +
                             self.b_in)

    @staticmethod
    def _read(w_read, memory):
        # w_read : (nb_sample, memory_dim)
        # memory : (nb_sample, memory_dim, memory_width)
        # return dot(w_read, memory)

        return T.sum(w_read[:, :, None] * memory, axis=1)

    @staticmethod
    def _write(w_write, memory, erase, add):
        # w_write: (nb_sample, memory_dim)
        # memory : (nb_sample, memory_dim, memory_width)
        # erase/add: (nb_sample, memory_width)

        w_write  = w_write[:, :, None]
        erase    = erase[:, None, :]     # erase is a gate.
        add      = add[:, None, :]       # add is a bias

        # m_erased = memory * (1 - w_write * erase)
        # memory_t = m_erased + w_write * add  # (nb_sample, memory_dim, memory_width)
        memory_t = memory * (1 - w_write * erase) + \
                   add * w_write * (1 - erase)

        return memory_t

    def _step(self, input_t, mask_t,
              memory_tm1,
              w_write_tm1, w_read_tm1,
              controller_tm1):
        # input_t:     (nb_sample, input_dim)
        # memory_tm1:  (nb_sample, memory_dim, memory_width)
        # w_write_tm1: (nb_sample, memory_dim)
        # w_read_tm1:  (nb_sample, memory_dim)
        # controller_tm1: (nb_sample, hidden_dim)

        # read the memory
        if self.curr_input:
            info     = T.concatenate((controller_tm1, input_t), axis=1)
            w_read_t = self.reader(info, w_read_tm1, memory_tm1)
            read_tm1 = self._read(w_read_t, memory_tm1)
        else:
            read_tm1 = self._read(w_read_tm1, memory_tm1)       # (nb_sample, memory_width)

        # get the new controller (hidden states.)
        if self.recurrence:
            controller_t = self._controller(input_t, read_tm1, controller_tm1)
        else:
            controller_t = self._controller(input_t, read_tm1)  # (nb_sample, controller_size)

        # update the memory cell (if need)
        if not self.readonly:
            if self.curr_input:
                infow          = T.concatenate((controller_t, input_t), axis=1)
                w_write_t      = self.writer(infow, w_write_tm1, memory_tm1)     # (nb_sample, memory_dim)
                erase_t, add_t = self.writer.get_fixer(infow)                    # (nb_sample, memory_width)
            else:
                w_write_t      = self.writer(controller_t, w_write_tm1, memory_tm1)
                erase_t, add_t = self.writer.get_fixer(controller_t)
            memory_t           = self._write(w_write_t, memory_tm1, erase_t, add_t)  # (nb_sample, memory_dim, memory_width)
        else:
            w_write_t          = w_write_tm1
            memory_t           = memory_tm1

        # get the next reading weights.
        if not self.curr_input:
            w_read_t           = self.reader(controller_t, w_read_tm1, memory_t)  # (nb_sample, memory_dim)

        # over masking
        memory_t     = memory_t     * mask_t[:, :, None] + memory_tm1 * (1 - mask_t[:, :, None])
        w_read_t     = w_read_t     * mask_t + w_read_tm1     * (1 - mask_t)
        w_write_t    = w_write_t    * mask_t + w_write_tm1    * (1 - mask_t)
        controller_t = controller_t * mask_t + controller_tm1 * (1 - mask_t)

        return memory_t, w_write_t, w_read_t, controller_t

    def __call__(self, X, mask=None, M=None, init_ww=None,
                 init_wr=None, init_c=None, return_sequence=False,
                 one_step=False, return_full=False):
        # recurrent cell only work for tensor.
        if X.ndim == 2:
            X = X[:, None, :]
        nb_samples = X.shape[0]

        # mask
        if mask is None:
            mask = T.alloc(1., X.shape[0], 1)

        padded_mask = self.get_padded_shuffled_mask(mask, pad=0)
        X = X.dimshuffle((1, 0, 2))

        # ***********************************************************************
        # initialization states
        if M is None:
            memory_init  = T.repeat(self.memory_init[None, :, :], nb_samples, axis=0)
        else:
            memory_init  = M

        if init_wr is None:
            w_read_init  = T.repeat(self.w_read_init, nb_samples, axis=0)
        else:
            w_read_init  = init_wr

        if init_ww is None:
            w_write_init = T.repeat(self.w_write_init, nb_samples, axis=0)
        else:
            w_write_init = init_ww

        if init_c is None:
            contr_init   = T.repeat(self.contr_init, nb_samples, axis=0)
        else:
            contr_init   = init_c
        # ************************************************************************

        outputs_info = [memory_init, w_write_init, w_read_init, contr_init]

        if one_step:
            seq = [X[0], padded_mask[0]]
            outputs = self._step(*(seq + outputs_info))
            return outputs
        else:
            seq = [X, padded_mask]
            outputs, _ = theano.scan(
                self._step,
                sequences=seq,
                outputs_info=outputs_info,
                name='controller_recurrence'
            )

        self.monitor['memory_info'] = outputs

        if not return_full:
            if return_sequence:
                return outputs[-1].dimshuffle((1, 0, 2))
            return outputs[-1][-1]
        else:
            if return_sequence:
                return [a.dimshuffle((1, 0, 2)) for a in outputs]
            return [a[-1] for a in outputs]