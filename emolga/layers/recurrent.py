# -*- coding: utf-8 -*-
from abc import abstractmethod
from .core import *


class Recurrent(MaskedLayer):
    """
        Recurrent Neural Network
    """

    @staticmethod
    def get_padded_shuffled_mask(mask, pad=0):
        """
        change the order of dims of mask, to match the dim of inputs outside
            [1] change the 2D matrix into 3D, (nb_samples, max_sent_len, 1)
            [2] dimshuffle to (max_sent_len, nb_samples, 1)
            the value on dim=0 could be either 0 or 1?
        :param: mask, shows x is a word (!=0) or not(==0), shape=(n_samples, max_sent_len)
        """
        # mask is (n_samples, time)
        assert mask, 'mask cannot be None'
        # pad a dim of 1 to the right, (nb_samples, max_sent_len, 1)
        mask = T.shape_padright(mask)
        # mask = T.addbroadcast(mask, -1), make the new dim broadcastable
        mask = T.addbroadcast(mask, mask.ndim-1)

        # change the order of dims, to match the dim of inputs outside
        mask = mask.dimshuffle(1, 0, 2)  # (max_sent_len, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')


class GRU(Recurrent):
    """
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatio-temporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        z_t         = tanh(W_z*x + U_z*h_t-1 + b_z)
        r_t         = tanh(W_r*x + U_r*h_t-1 + b_r)
        hh_t        = tanh(W_h*x + U_r*(r_t*h_t-1) + b_h)
        h_t         = z_t * h_t-1 + (1 - z_t) * hh_t

        The doc product computation regarding x is independent from time
            so it could be done out of the recurrent process (in advance)
                x_z         = dot(X, self.W_z, self.b_z)
                x_r         = dot(X, self.W_r, self.b_r)
                x_h         = dot(X, self.W_h, self.b_h)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    """

    def __init__(self,
                 input_dim,
                 output_dim=128,
                 context_dim=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 name=None, weights=None):

        super(GRU, self).__init__()
        """
        Standard GRU.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        # W is a matrix to map input x_t
        self.W_z = self.init((self.input_dim, self.output_dim))
        self.W_r = self.init((self.input_dim, self.output_dim))
        self.W_h = self.init((self.input_dim, self.output_dim))
        # U is a matrix to map hidden state of last time h_t-1
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        # bias terms
        self.b_z = shared_zeros(self.output_dim)
        self.b_r = shared_zeros(self.output_dim)
        self.b_h = shared_zeros(self.output_dim)

        # set names
        self.W_z.name, self.U_z.name, self.b_z.name = 'Wz', 'Uz', 'bz'
        self.W_r.name, self.U_r.name, self.b_r.name = 'Wr', 'Ur', 'br'
        self.W_h.name, self.U_h.name, self.b_h.name = 'Wh', 'Uh', 'bh'

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        """
        GRU with context inputs.
        """
        if context_dim is not None:
            self.context_dim = context_dim
            self.C_z = self.init((self.context_dim, self.output_dim))
            self.C_r = self.init((self.context_dim, self.output_dim))
            self.C_h = self.init((self.context_dim, self.output_dim))
            self.C_z.name, self.C_r.name, self.C_h.name = 'Cz', 'Cr', 'Ch'

            self.params += [self.C_z, self.C_r, self.C_h]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_t,
              h_tm1,
              u_z, u_r, u_h):
        """
        One step computation of GRU for a batch of data at time t
                sequences=[x_z, x_r, x_h, padded_mask],
                outputs_info=init_h,
                non_sequences=[self.U_z, self.U_r, self.U_h]
        :param xz_t, xr_t, xh_t:
                        value of x of time t after gate z/r/h (computed beforehand)
                            shape=(n_samples, output_emb_dim)
        :param mask_t:  mask of time t, indicates whether t-th token is a word, shape=(n_samples, 1)
        :param h_tm1:   hidden value (output) of last time, shape=(nb_samples, output_emb_dim)
        :param u_z, u_r, u_h:
                        mapping matrix for hidden state of time t-1
                        shape=(output_emb_dim, output_emb_dim)
        :return: h_t:   output, hidden state of time t, shape=(nb_samples, output_emb_dim)
        """
        # h_mask_tm1 = mask_tm1 * h_tm1
        # Here we use a GroundHog-like style which allows

        # activation value of update/reset gate, shape=(n_samples, 1)
        z          = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r          = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t       = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t        = z * h_tm1 + (1 - z) * hh_t

        # why use mask_t to mix up h_t and h_tm1 again?
        #   if current term is None (padding term, mask=0), then drop the update (0*h_t and keep use the last state(1*h_tm1)
        h_t        = mask_t * h_t + (1 - mask_t) * h_tm1
        return h_t

    def _step_gate(self,
                   xz_t, xr_t, xh_t, mask_t,
                   h_tm1,
                   u_z, u_r, u_h):
        """
        One step computation of GRU
        :returns
            h_t:   output, hidden state of time t, shape=(n_samples, output_emb_dim)
            z:     value of update gate (after activation), shape=(n_samples, 1)
            r:     value of reset gate (after activation), shape=(n_samples, 1)
        """
        # h_mask_tm1 = mask_tm1 * h_tm1
        # Here we use a GroundHog-like style which allows
        z          = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r          = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t       = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t        = z * h_tm1 + (1 - z) * hh_t
        h_t        = mask_t * h_t + (1 - mask_t) * h_tm1
        return h_t, z, r

    def __call__(self, X, mask=None, C=None, init_h=None,
                 return_sequence=False, one_step=False,
                 return_gates=False):
        """
        :param X:       input sequence, a list of word vectors, shape=(n_samples, max_sent_len, input_emb_dim)
        :param mask:    input mask, shows x is a word (!=0) or not(==0), shape=(n_samples, max_sent_len)
        :param C:       context, for encoder is none
        :param init_h:  initial hidden state
        :param return_sequence: if True, return the encoding at each time, or only return the end state
        :param one_step: only go one step computation, or will be done by theano.scan()
        :param return_gates: whether return the gate state
        :return:
        """
        # recurrent cell only work for tensor
        if X.ndim == 2: # X.ndim == 3, shape=(n_samples, max_sent_len, input_emb_dim)
            X = X[:, None, :]
            if mask is not None:
                mask = mask[:, None]

        # mask, shape=(n_samples, max_sent_len)
        if mask is None:  # sampling or beam-search
            mask = T.alloc(1., X.shape[0], 1)

        # one step
        if one_step:
            assert init_h, 'previous state must be provided!'

        # reshape the mask to shape=(max_sent_len, n_samples, 1)
        padded_mask = self.get_padded_shuffled_mask(mask, pad=0)
        X           = X.dimshuffle((1, 0, 2))     # X:   (max_sent_len, nb_samples, input_emb_dim)
        # compute the gate values at each time in advance
        #       shape of W = (input_emb_dim, output_emb_dim)
        x_z         = dot(X, self.W_z, self.b_z)  # x_z: (max_sent_len, nb_samples, output_emb_dim)
        x_r         = dot(X, self.W_r, self.b_r)  # x_r: (max_sent_len, nb_samples, output_emb_dim)
        x_h         = dot(X, self.W_h, self.b_h)  # x_h: (max_sent_len, nb_samples, output_emb_dim)

        """
        GRU with constant context. (no attention here.)
        """
        if C is not None:
            assert C.ndim == 2
            ctx_step = C.dimshuffle('x', 0, 1)    # C: (nb_samples, context_dim)
            x_z     += dot(ctx_step, self.C_z)
            x_r     += dot(ctx_step, self.C_r)
            x_h     += dot(ctx_step, self.C_h)

        """
        GRU with additional initial/previous state.
        """
        if init_h is None:
            init_h = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if not return_gates:
            if one_step:
                seq          = [x_z, x_r, x_h, padded_mask]    # A hidden BUG (1)+++(1) !?!!!?!!?!?
                outputs_info = [init_h]
                non_seq      = [self.U_z, self.U_r, self.U_h]
                outputs = self._step(*(seq + outputs_info + non_seq))

            else:
                outputs, _ = theano.scan(
                    self._step,
                    sequences=[x_z, x_r, x_h, padded_mask],
                    outputs_info=init_h,
                    non_sequences=[self.U_z, self.U_r, self.U_h]
                )

            # return hidden state of all times, shape=(nb_samples, max_sent_len, input_emb_dim)
            if return_sequence:
                return outputs.dimshuffle((1, 0, 2))
            # hidden state of last time, shape=(nb_samples, output_emb_dim)
            return outputs[-1]
        else:
            if one_step:
                seq             = [x_z, x_r, x_h, padded_mask]    # A hidden BUG (1)+++(1) !?!!!?!!?!?
                outputs_info    = [init_h]
                non_seq         = [self.U_z, self.U_r, self.U_h]
                outputs, zz, rr = self._step_gate(*(seq + outputs_info + non_seq))

            else:
                outputx, _ = theano.scan(
                    self._step_gate,
                    sequences=[x_z, x_r, x_h, padded_mask],
                    outputs_info=[init_h, None, None],
                    non_sequences=[self.U_z, self.U_r, self.U_h]
                )
                outputs, zz, rr = outputx

            if return_sequence:
                return outputs.dimshuffle((1, 0, 2)), zz.dimshuffle((1, 0, 2)), rr.dimshuffle((1, 0, 2))
            return outputs[-1], zz[-1], rr[-1]


class JZS3(Recurrent):
    """
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT3` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """
    def __init__(self,
                 input_dim,
                 output_dim=128,
                 context_dim=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 name=None, weights=None):

        super(JZS3, self).__init__()
        """
        Standard model
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros(self.output_dim)

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros(self.output_dim)

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros(self.output_dim)

        # set names
        self.W_z.name, self.U_z.name, self.b_z.name = 'Wz', 'Uz', 'bz'
        self.W_r.name, self.U_r.name, self.b_r.name = 'Wr', 'Ur', 'br'
        self.W_h.name, self.U_h.name, self.b_h.name = 'Wh', 'Uh', 'bh'

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        """
        context inputs.
        """
        if context_dim is not None:
            self.context_dim = context_dim
            self.C_z = self.init((self.context_dim, self.output_dim))
            self.C_r = self.init((self.context_dim, self.output_dim))
            self.C_h = self.init((self.context_dim, self.output_dim))
            self.C_z.name, self.C_r.name, self.C_h.name = 'Cz', 'Cr', 'Ch'

            self.params += [self.C_z, self.C_r, self.C_h]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_t,
              h_tm1,
              u_z, u_r, u_h):
        # h_mask_tm1 = mask_tm1 * h_tm1
        z     = self.inner_activation(xz_t + T.dot(T.tanh(h_tm1), u_z))
        r     = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t  = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t   = (hh_t * z + h_tm1 * (1 - z)) * mask_t + (1 - mask_t) * h_tm1
        return h_t

    def __call__(self, X, mask=None, C=None, init_h=None, return_sequence=False, one_step=False):
        # recurrent cell only work for tensor
        if X.ndim == 2:
            X = X[:, None, :]

        # mask
        if mask is None:  # sampling or beam-search
            mask = T.alloc(1., X.shape[0], X.shape[1])

        # one step
        if one_step:
            assert init_h, 'previous state must be provided!'

        padded_mask = self.get_padded_shuffled_mask(mask, pad=0)
        X = X.dimshuffle((1, 0, 2))

        x_z = dot(X, self.W_z, self.b_z)
        x_r = dot(X, self.W_r, self.b_r)
        x_h = dot(X, self.W_h, self.b_h)

        """
        JZS3 with constant context. (not attention here.)
        """
        if C is not None:
            assert C.ndim == 2
            ctx_step = C.dimshuffle('x', 0, 1)    # C: (nb_samples, context_dim)
            x_z     += dot(ctx_step, self.C_z)
            x_r     += dot(ctx_step, self.C_r)
            x_h     += dot(ctx_step, self.C_h)

        """
        JZS3 with additional initial/previous state.
        """
        if init_h is None:
            init_h = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if one_step:
            seq          = [x_z, x_r, x_h, padded_mask]
            outputs_info = [init_h]
            non_seq      = [self.U_z, self.U_r, self.U_h]
            outputs = self._step(*(seq + outputs_info + non_seq))

        else:
            outputs, updates = theano.scan(
                self._step,
                sequences=[x_z, x_r, x_h, padded_mask],
                outputs_info=init_h,
                non_sequences=[self.U_z, self.U_r, self.U_h],
            )

        if return_sequence:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]


class LSTM(Recurrent):
    def __init__(self,
                 input_dim=0,
                 output_dim=128,
                 context_dim=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid',
                 name=None, weights=None):

        super(LSTM, self).__init__()
        """
        Standard model
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        # input gate param.
        self.W_i = self.init((self.input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros(self.output_dim)

        # forget gate param.
        self.W_f = self.init((self.input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init(self.output_dim)  # forget gate needs one bias.

        # output gate param.
        self.W_o = self.init((self.input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros(self.output_dim)

        # memory param.
        self.W_c = self.init((self.input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros(self.output_dim)

        # set names
        self.W_i.name, self.U_i.name, self.b_i.name = 'Wi', 'Ui', 'bi'
        self.W_f.name, self.U_f.name, self.b_f.name = 'Wf', 'Uf', 'bf'
        self.W_o.name, self.U_o.name, self.b_o.name = 'Wo', 'Uo', 'bo'
        self.W_c.name, self.U_c.name, self.b_c.name = 'Wc', 'Uc', 'bc'

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_c, self.U_c, self.b_c,
        ]

        """
        context inputs.
        """
        if context_dim is not None:
            self.context_dim = context_dim
            self.C_i = self.init((self.context_dim, self.output_dim))
            self.C_f = self.init((self.context_dim, self.output_dim))
            self.C_o = self.init((self.context_dim, self.output_dim))
            self.C_c = self.init((self.context_dim, self.output_dim))
            self.C_i.name, self.C_f.name, self.C_o.name, self.C_c.name = 'Ci', 'Cf', 'Co', 'Cc'

            self.params += [self.C_i, self.C_f, self.C_o, self.C_c]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_t,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        # h_mask_tm1 = mask_tm1 * h_tm1

        i     = self.inner_activation(xi_t + T.dot(h_tm1, u_i))  # input  gate
        f     = self.inner_activation(xf_t + T.dot(h_tm1, u_f))  # forget gate
        o     = self.inner_activation(xo_t + T.dot(h_tm1, u_o))  # output gate
        c     = self.activation(xc_t + T.dot(h_tm1, u_c))        # memory updates

        # update the memory cell.
        c_t   = f * c_tm1 + i * c
        h_t   = o * self.activation(c_t)

        # masking
        c_t   = c_t * mask_t + (1 - mask_t) * c_tm1
        h_t   = h_t * mask_t + (1 - mask_t) * h_tm1
        return h_t, c_t

    def input_embed(self, X, C=None):
        x_i = dot(X, self.W_i, self.b_i)
        x_f = dot(X, self.W_f, self.b_f)
        x_o = dot(X, self.W_o, self.b_o)
        x_c = dot(X, self.W_c, self.b_c)

        """
        LSTM with constant context. (not attention here.)
        """
        if C is not None:
            assert C.ndim == 2
            ctx_step = C.dimshuffle('x', 0, 1)    # C: (nb_samples, context_dim)
            x_i     += dot(ctx_step, self.C_i)
            x_f     += dot(ctx_step, self.C_f)
            x_o     += dot(ctx_step, self.C_o)
            x_c     += dot(ctx_step, self.C_c)

        return x_i, x_f, x_o, x_c

    def __call__(self, X, mask=None, C=None, init_h=None, init_c=None, return_sequence=False, one_step=False):
        # recurrent cell only work for tensor
        if X.ndim == 2:
            X = X[:, None, :]

        # mask
        if mask is None:  # sampling or beam-search
            mask = T.alloc(1., X.shape[0], X.shape[1])

        # one step
        if one_step:
            assert init_h, 'previous state must be provided!'

        padded_mask = self.get_padded_shuffled_mask(mask, pad=0)
        X = X.dimshuffle((1, 0, 2))
        x_i, x_f, x_o, x_c = self.input_embed(X, C)

        """
        LSTM with additional initial/previous state.
        """
        if init_h is None:
            init_h = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if init_c is None:
            init_c = init_h

        if one_step:
            seq          = [x_i, x_f, x_o, x_c, padded_mask]
            outputs_info = [init_h, init_c]
            non_seq      = [self.U_i, self.U_f, self.U_o, self.U_c]
            outputs = self._step(*(seq + outputs_info + non_seq))

        else:
            outputs, updates = theano.scan(
                self._step,
                sequences=[x_i, x_f, x_o, x_c, padded_mask],
                outputs_info=[init_h, init_c],
                non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            )

        if return_sequence:
            return outputs[0].dimshuffle((1, 0, 2)), outputs[1].dimshuffle((1, 0, 2))  # H, C
        return outputs[0][-1], outputs[1][-1]


