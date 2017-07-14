__author__ = 'jiataogu'
"""
The file is the implementation of Grid-LSTM
In this stage we only support 2D LSTM with Pooling.
"""
from recurrent import *
from attention import Attention
import logging
import copy
logger = logging.getLogger(__name__)


class Grid(Recurrent):
    """
    Grid Cell for Grid-LSTM
    ===================================================
    LSTM
            [h', m'] = LSTM(x, h, m):
                gi = sigmoid(Wi * x + Ui * h + Vi * m)  # Vi is peep-hole
                gf = sigmoid(Wf * x + Uf * h + Vf * m)
                go = sigmoid(Wo * x + Uo * h + Vo * m)
                gc = tanh(Wc * x +Uc * h)

                m' = gf @ m + gi @ gc  (@ represents element-wise dot.)
                h' = go @ tanh(m')

    ===================================================
    Grid
    (here is an example for 2D Grid LSTM with priority dimension = 1)
     -------------
    |    c'  d'   |     Grid Block and Grid Updates.
    | a         a'|
    |             |     [d' c'] = LSTM_d([b, d],  c)
    | b         b'|     [a' b'] = LSTM_t([b, d'], a)
    |    c   d    |
     -------------
    ===================================================
    Details please refer to:
        "Grid Long Short-Term Memory", http://arxiv.org/abs/1507.01526
    """
    def __init__(self,
                 output_dims,
                 input_dims,    # [0, ... 0], 0 represents no external inputs.
                 priority=1,
                 peephole=True,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid',
                 use_input=False,
                 name=None, weights=None,
                 identity_connect=None
                 ):
        super(Grid, self).__init__()

        assert len(output_dims) == 2, 'in this stage, we only support 2D Grid-LSTM'
        assert len(input_dims)  == len(output_dims), '# of inputs must match # of outputs.'

        """
        Initialization.
        """
        self.input_dims       = input_dims
        self.output_dims      = output_dims
        self.N                = len(output_dims)
        self.priority         = priority
        self.peephole         = peephole
        self.use_input        = use_input

        self.init             = initializations.get(init)
        self.inner_init       = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation       = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.identity_connect = identity_connect
        self.axies            = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}  # only support at most 4D now!

        """
        Others info.
        """
        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def build(self):
        """
        Build the model weights
        """
        logger.info("Building GridPool-LSTM !!")
        self.W = dict()
        self.U = dict()
        self.V = dict()
        self.b = dict()

        # ******************************************************************************************
        for k in xrange(self.N):       # N-Grids (for 2 dimensions, 0 is for time; 1 is for depth.)
            axis  = self.axies[k]
            # input layers:
            if self.input_dims[k] > 0 and self.use_input:
                # use the data information.
                self.W[axis + '#i'], self.W[axis + '#f'], \
                self.W[axis + '#o'], self.W[axis + '#c']  \
                      = [self.init((self.input_dims[k], self.output_dims[k])) for _ in xrange(4)]

            # hidden layers:
            for j in xrange(self.N):   # every hidden states inputs.
                pos   = self.axies[j]
                if k == j:
                    self.U[axis + pos + '#i'], self.U[axis + pos + '#f'], \
                    self.U[axis + pos + '#o'], self.U[axis + pos + '#c']  \
                        = [self.inner_init((self.output_dims[j], self.output_dims[k])) for _ in xrange(4)]
                else:
                    self.U[axis + pos + '#i'], self.U[axis + pos + '#f'], \
                    self.U[axis + pos + '#o'], self.U[axis + pos + '#c']  \
                        = [self.init((self.output_dims[j], self.output_dims[k])) for _ in xrange(4)]

            # bias layers:
            self.b[axis + '#i'], self.b[axis + '#o'], self.b[axis + '#c']  \
                      = [shared_zeros(self.output_dims[k]) for _ in xrange(3)]
            self.b[axis + '#f'] = self.forget_bias_init(self.output_dims[k])

            # peep-hole layers:
            if self.peephole:
                self.V[axis + '#i'], self.V[axis + '#f'], self.V[axis + '#o'] \
                      = [self.init(self.output_dims[k]) for _ in xrange(3)]
        # *****************************************************************************************

        # set names for these weights
        for A, n in zip([self.W, self.U, self.b, self.V], ['W', 'U', 'b', 'V']):
            for w in A:
                A[w].name = n + '_' + w

        # set parameters
        self.params = [self.W[s] for s in self.W] + \
                      [self.U[s] for s in self.U] + \
                      [self.b[s] for s in self.b] + \
                      [self.V[s] for s in self.V]

    def lstm_(self, k, H, m, x, identity=False):
        """
       LSTM
            [h', m'] = LSTM(x, h, m):
                gi = sigmoid(Wi * x + Ui * h + Vi * m)  # Vi is peep-hole
                gf = sigmoid(Wf * x + Uf * h + Vf * m)
                go = sigmoid(Wo * x + Uo * h + Vo * m)
                gc = tanh(Wc * x +Uc * h)

                m' = gf @ m + gi @ gc  (@ represents element-wise dot.)
                h' = go @ tanh(m')

        """
        assert len(H) == self.N, 'we have to use all the hidden states in Grid LSTM'
        axis           = self.axies[k]

        # *************************************************************************
        # bias energy
        ei, ef, eo, ec = [self.b[axis + p] for p in ['#i', '#f', '#o', '#c']]

        # hidden energy
        for j in xrange(self.N):
            pos  = self.axies[j]

            ei  += T.dot(H[j], self.U[axis + pos + '#i'])
            ef  += T.dot(H[j], self.U[axis + pos + '#f'])
            eo  += T.dot(H[j], self.U[axis + pos + '#o'])
            ec  += T.dot(H[j], self.U[axis + pos + '#c'])

        # input energy (if any)
        if self.input_dims[k] > 0 and self.use_input:
            ei  += T.dot(x, self.W[axis + '#i'])
            ef  += T.dot(x, self.W[axis + '#f'])
            eo  += T.dot(x, self.W[axis + '#o'])
            ec  += T.dot(x, self.W[axis + '#c'])

        # peep-hole connections
        if self.peephole:
            ei  += m * self.V[axis + '#i'][None, :]
            ef  += m * self.V[axis + '#f'][None, :]
            eo  += m * self.V[axis + '#o'][None, :]
        # *************************************************************************

        # compute the gates.
        i        = self.inner_activation(ei)
        f        = self.inner_activation(ef)
        o        = self.inner_activation(eo)
        c        = self.activation(ec)

        # update the memory and hidden states.
        m_new    = f * m + i * c
        h_new    = o * self.activation(m_new)

        return h_new, m_new

    def grid_(self,
              hs_i,
              ms_i,
              xs_i,
              priority=1,
              identity=None):
        """
        ===================================================
        Grid (2D as an example)
         -------------
        |    c'  d'   |     Grid Block and Grid Updates.
        | a         a'|
        |             |     [d' c'] = LSTM_d([b, d],  c)
        | b         b'|     [a' b'] = LSTM_t([b, d'], a)   priority
        |    c   d    |
         -------------
         a = my | b = hy | c = mx | d = hx
        ===================================================

        Currently masking is not considered in GridLSTM.
        """
        # compute LSTM updates for non-priority dimensions
        H_new   = hs_i
        M_new   = ms_i
        for k in xrange(self.N):
            if k == priority:
                continue
            m   = ms_i[k]
            x   = xs_i[k]
            H_new[k], M_new[k] \
                = self.lstm_(k, hs_i, m, x)

            if identity is not None:
                if identity[k]:
                    H_new[k] += hs_i[k]

        # compute LSTM updates along the priority dimension
        if priority >= 0:
            hs_ii   = H_new
            H_new[priority], M_new[priority] \
                    = self.lstm_(priority, hs_ii, ms_i[priority], xs_i[priority])
            if identity is not None:
                if identity[priority]:
                    H_new[priority] += hs_ii[priority]

        return H_new, M_new


class SequentialGridLSTM(Grid):
    """
    Details please refer to:
        "Grid Long Short-Term Memory",
            http://arxiv.org/abs/1507.01526

    SequentialGridLSTM is a typical 2D-GridLSTM,
    which has one flexible dimension (time) and one fixed dimension (depth)
    Input information is added along x-axis.
    """
    def __init__(self,
                 # parameters for Grid.
                 output_dims,
                 input_dims,    # [0, ... 0], 0 represents no external inputs.
                 priority=1,
                 peephole=True,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid',
                 use_input=False,
                 name=None, weights=None,
                 identity_connect=None,

                 # parameters for 2D-GridLSTM
                 depth=5,
                 learn_init=False,
                 pooling=True,
                 attention=False,
                 shared=True,
                 dropout=0,
                 rng=None,
                 ):
        super(Grid, self).__init__()

        assert len(output_dims) == 2, 'in this stage, we only support 2D Grid-LSTM'
        assert len(input_dims)  == len(output_dims), '# of inputs must match # of outputs.'
        assert input_dims[1]    == 0, 'we have no y-axis inputs here.'
        assert shared, 'we share the weights in this stage.'
        assert not (attention and pooling), 'attention and pooling cannot be set at the same time.'

        """
        Initialization.
        """
        logger.info(":::: Sequential Grid-Pool LSTM ::::")
        self.input_dims       = input_dims
        self.output_dims      = output_dims
        self.N                = len(output_dims)
        self.depth            = depth
        self.dropout          = dropout

        self.priority         = priority
        self.peephole         = peephole
        self.use_input        = use_input
        self.pooling          = pooling
        self.attention        = attention
        self.learn_init       = learn_init

        self.init             = initializations.get(init)
        self.inner_init       = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation       = activations.get(activation)
        self.relu             = activations.get('relu')
        self.inner_activation = activations.get(inner_activation)

        self.identity_connect = identity_connect
        self.axies            = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}  # only support at most 4D now!

        if self.identity_connect is not None:
            logger.info('Identity Connection: {}'.format(self.identity_connect))

        """
        Build the model weights.
        """
        # build the centroid grid.
        self.build()

        # input projection layer (projected to time-axis)       [x]
        self.Ph  = Dense(input_dims[0], output_dims[0], name='Ph')
        self.Pm  = Dense(input_dims[0], output_dims[0], name='Pm')

        self._add(self.Ph)
        self._add(self.Pm)

        # learn init for depth-axis hidden states/memory cells. [y]
        if self.learn_init:
            self.M0      = self.init((depth, output_dims[1]))
            if self.pooling:
                self.H0  = self.init(output_dims[1])
            else:
                self.H0  = self.init((depth, output_dims[1]))

            self.M0.name, self.H0.name = 'M0', 'H0'
            self.params += [self.M0, self.H0]

        # if we use attention instead of max-pooling
        if self.pooling:
            self.PP      = Dense(output_dims[1] + input_dims[0], output_dims[1], # init='orthogonal',
                                 name='PP', activation='linear')
            self._add(self.PP)

        if self.attention:
            self.A       = Attention(target_dim=input_dims[0],
                                     source_dim=output_dims[1],
                                     hidden_dim=200, name='attender')
            self._add(self.A)

        # if self.dropout > 0:
        #     logger.info(">>>>>> USE DropOut !! <<<<<<")
        #     self.D       = Dropout(rng=rng, p=self.dropout, name='Dropout')

        """
        Others info.
        """
        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def _step(self, *args):
        # since depth is not determined, we cannot decide the number of inputs
        # for one time step.
        # if pooling is True:
        #    args = [raw_input] +       (sequence)
        #           [hy] + [my]*depth   (output_info)
        #
        inputs = args[0]
        Hy_tm1 = [args[k] for k in range(1, 1 + self.depth)]
        My_tm1 = [args[k] for k in range(1 + self.depth, 1 + 2 * self.depth)]

        # x_axis input projection (get hx_t, mx_t)
        hx_t   = self.Ph(inputs)           # (nb_samples, output_dim0)
        mx_t   = self.Pm(inputs)           # (nb_samples, output_dim0)

        # build computation path from bottom to top.
        Hx_t   = [hx_t]
        Mx_t   = [mx_t]
        Hy_t   = []
        My_t   = []
        for d in xrange(self.depth):
            hs_i       = [Hx_t[-1], Hy_tm1[d]]
            ms_i       = [Mx_t[-1], My_tm1[d]]
            xs_i       = [inputs,   T.zeros_like(inputs)]

            hs_o, ms_o = self.grid_(hs_i, ms_i, xs_i, priority=self.priority, identity=self.identity_connect)

            Hx_t      += [hs_o[0]]
            Hy_t      += [hs_o[1]]
            Mx_t      += [ms_o[0]]
            My_t      += [ms_o[1]]

        hx_out = Hx_t[-1]
        mx_out = Mx_t[-1]

        # get the output (output_y, output_x)
        # MAX-Pooling
        if self.pooling:
            # hy_t       = T.max([self.PP(hy) for hy in Hy_t], axis=0)
            hy_t       = T.max([self.PP(T.concatenate([hy, inputs], axis=-1)) for hy in Hy_t], axis=0)
            Hy_t       = [hy_t] * self.depth

        if self.attention:
            HHy_t      = T.concatenate([hy[:, None, :] for hy in Hy_t], axis=1)  # (nb_samples, n_depth, out_dim1)
            annotation = self.A(inputs, HHy_t)   # (nb_samples, n_depth)
            hy_t       = T.sum(HHy_t * annotation[:, :, None], axis=1)           # (nb_samples, out_dim1)
            Hy_t       = [hy_t] * self.depth

        R = Hy_t + My_t + [hx_out, mx_out]
        return tuple(R)

    def __call__(self, X, init_H=None, init_M=None,
                 return_sequence=False, one_step=False,
                 return_info='hy', train=True):
        # It is training/testing path
        self.train = train

        # recently we did not support masking.
        if X.ndim == 2:
            X = X[:, None, :]

        # one step
        if one_step:
            assert init_H is not None, 'previous state must be provided!'
            assert init_M is not None, 'previous cell must be provided!'

        X = X.dimshuffle((1, 0, 2))
        if init_H is None:
            if self.learn_init:
                init_m     = T.repeat(self.M0[:, None, :], X.shape[1], axis=1)
                if self.pooling:
                    init_h = T.repeat(self.H0[None, :], self.depth, axis=0)
                else:
                    init_h = self.H0
                init_h     = T.repeat(init_h[:, None, :], X.shape[1], axis=1)

                init_H     = []
                init_M     = []
                for j in xrange(self.depth):
                    init_H.append(init_h[j])
                    init_M.append(init_m[j])
            else:
                init_H     = [T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dims[1]), 1)] * self.depth
                init_M     = [T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dims[1]), 1)] * self.depth
            pass

        # computational graph !
        if not one_step:
            sequences    = [X]
            outputs_info = init_H + init_M + [None, None]
            outputs, _   = theano.scan(
                self._step,
                sequences=sequences,
                outputs_info=outputs_info
            )
        else:
            outputs      = self._step(*([X[0]] + init_H + init_M))

        if   return_info == 'hx':
            if return_sequence:
                return outputs[0].dimshuffle((1, 0, 2))
            return outputs[-2][-1]
        elif return_info == 'hy':
            assert self.pooling or self.attention, 'y-axis hidden states are only used in the ``Pooling Mode".'
            if return_sequence:
                return outputs[2].dimshuffle((1, 0, 2))
            return outputs[2][-1]
        elif return_info == 'hxhy':
            assert self.pooling or self.attention, 'y-axis hidden states are only used in the ``Pooling Mode".'
            if return_sequence:
                return outputs[-2].dimshuffle((1, 0, 2)), outputs[2].dimshuffle((1, 0, 2))    # x-y
            return outputs[-2][-1], outputs[2][-1]


class PyramidGridLSTM2D(Grid):
    """
    A variant version of Sequential LSTM where we introduce a Pyramid structure.
    """
    def __init__(self,
                 # parameters for Grid.
                 output_dims,
                 input_dims,    # [0, ... 0], 0 represents no external inputs.
                 priority=1,
                 peephole=True,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid',
                 use_input=True,
                 name=None, weights=None,
                 identity_connect=None,

                 # parameters for 2D-GridLSTM
                 depth=5,
                 learn_init=False,
                 shared=True,
                 dropout=0
                 ):

        super(Grid, self).__init__()
        assert len(output_dims) == 2, 'in this stage, we only support 2D Grid-LSTM'
        assert len(input_dims)  == len(output_dims), '# of inputs must match # of outputs.'
        assert output_dims[0] == output_dims[1], 'Here we only support square model.'
        assert shared, 'we share the weights in this stage.'
        assert use_input, 'use input and add them in the middle'

        """
        Initialization.
        """
        logger.info(":::: Sequential Grid-Pool LSTM ::::")
        self.input_dims       = input_dims
        self.output_dims      = output_dims
        self.N                = len(output_dims)
        self.depth            = depth
        self.dropout          = dropout

        self.priority         = priority
        self.peephole         = peephole
        self.use_input        = use_input
        self.learn_init       = learn_init

        self.init             = initializations.get(init)
        self.inner_init       = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation       = activations.get(activation)
        self.relu             = activations.get('relu')
        self.inner_activation = activations.get(inner_activation)

        self.identity_connect = identity_connect
        self.axies            = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}  # only support at most 4D now!

        """
        Build the model weights.
        """
        # build the centroid grid.
        self.build()

        # # input projection layer (projected to time-axis)       [x]
        # self.Ph  = Dense(input_dims[0], output_dims[0], name='Ph')
        # self.Pm  = Dense(input_dims[0], output_dims[0], name='Pm')
        #
        # self._add(self.Ph)
        # self._add(self.Pm)

        # learn init/
        if self.learn_init:
            self.hx0 = self.init((1, output_dims[0]))
            self.hy0 = self.init((1, output_dims[1]))
            self.mx0 = self.init((1, output_dims[0]))
            self.my0 = self.init((1, output_dims[1]))

            self.hx0.name, self.hy0.name = 'hx0', 'hy0'
            self.mx0.name, self.my0.name = 'mx0', 'my0'
            self.params += [self.hx0, self.hy0, self.mx0, self.my0]

        """
        Others info.
        """
        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def _step(self, *args):
        inputs = args[0]
        hx_tm1 = args[1]
        mx_tm1 = args[2]
        hy_tm1 = args[3]
        my_tm1 = args[4]

        # zero constant inputs.
        pre_info    = [[[T.zeros_like(hx_tm1)
                         for _ in xrange(self.depth)]
                         for _ in xrange(self.depth)]
                         for _ in xrange(4)]  # hx, mx, hy, my

        pre_inputs  = [[T.zeros_like(inputs)
                       for _ in xrange(self.depth)]
                       for _ in xrange(self.depth)]

        for kk in xrange(self.depth):
            pre_inputs[kk][kk] = inputs

        pre_info[0][0][0] = hx_tm1
        pre_info[1][0][0] = mx_tm1
        pre_info[2][0][0] = hy_tm1
        pre_info[3][0][0] = my_tm1

        for step_x in xrange(self.depth):
            for step_y in xrange(self.depth):
                # input hidden/memory/input information
                print pre_info[0][-1][-1], pre_info[2][-1][-1]

                hs_i  = [pre_info[0][step_x][step_y],
                         pre_info[2][step_x][step_y]]
                ms_i  = [pre_info[1][step_x][step_y],
                         pre_info[3][step_x][step_y]]
                xs_i  = [pre_inputs[step_x][step_y],
                         pre_inputs[step_x][step_y]]

                # compute grid-lstm
                hs_o, ms_o = self.grid_(hs_i, ms_i, xs_i, priority =-1)

                # output hidden/memory information
                if (step_x == self.depth - 1) and (step_y == self.depth - 1):
                    hx_t, mx_t, hy_t, my_t = hs_o[0], ms_o[0], hs_o[1], ms_o[1]
                    return hx_t, mx_t, hy_t, my_t

                if step_x + 1 < self.depth:
                    pre_info[0][step_x + 1][step_y] = hs_o[0]
                    pre_info[1][step_x + 1][step_y] = ms_o[0]

                if step_y + 1 < self.depth:
                    pre_info[2][step_x][step_y + 1] = hs_o[1]
                    pre_info[3][step_x][step_y + 1] = ms_o[1]

    def __call__(self, X, init_x=None, init_y=None,
                 return_sequence=False, one_step=False):
        # recently we did not support masking.
        if X.ndim == 2:
            X = X[:, None, :]

        # one step
        if one_step:
            assert init_x is not None, 'previous x must be provided!'
            assert init_y is not None, 'previous y must be provided!'

        X = X.dimshuffle((1, 0, 2))
        if init_x is None:
            if self.learn_init:
                init_mx    = T.repeat(self.mx0, X.shape[1], axis=0)
                init_my    = T.repeat(self.my0, X.shape[1], axis=0)
                init_hx    = T.repeat(self.hx0, X.shape[1], axis=0)
                init_hy    = T.repeat(self.hy0, X.shape[1], axis=0)

                init_input = [init_hx, init_mx, init_hy, init_my]
            else:
                init_x     = [T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dims[0]), 1)] * 2
                init_y     = [T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dims[1]), 1)] * 2

                init_input = init_x + init_y
        else:
            init_input = init_x + init_y

        if not one_step:
            sequence       = [X]
            output_info    = init_input
            outputs, _     = theano.scan(
                self._step,
                sequences=sequence,
                outputs_info=output_info
            )
        else:
            outputs        = self._step(*([X[0]] + init_x + init_y))

        if return_sequence:
            hxs = outputs[0].dimshuffle((1, 0, 2))
            hys = outputs[2].dimshuffle((1, 0, 2))
            hs  = T.concatenate([hxs, hys], axis=-1)
            return hs
        else:
            hx  = outputs[0][-1]
            hy  = outputs[2][-1]
            h   = T.concatenate([hx, hy], axis=-1)
            return h


class PyramidLSTM(Layer):
    """
    A more flexible Pyramid LSTM structure!
    """
    def __init__(self,
                 # parameters for Grid.
                 output_dims,
                 input_dims,    # [0, ... 0], 0 represents no external inputs.
                 priority=1,
                 peephole=True,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid',
                 use_input=True,
                 name=None, weights=None,
                 identity_connect=None,

                 # parameters for 2D-GridLSTM
                 depth=5,
                 learn_init=False,
                 shared=True,
                 dropout=0
                 ):

        super(PyramidLSTM, self).__init__()
        assert len(output_dims) == 2, 'in this stage, we only support 2D Grid-LSTM'
        assert len(input_dims)  == len(output_dims), '# of inputs must match # of outputs.'
        assert output_dims[0] == output_dims[1], 'Here we only support square model.'
        assert shared, 'we share the weights in this stage.'
        assert use_input, 'use input and add them in the middle'

        """
        Initialization.
        """
        logger.info(":::: Sequential Grid-Pool LSTM ::::")
        self.N                = len(output_dims)
        self.depth            = depth
        self.dropout          = dropout

        self.priority         = priority
        self.peephole         = peephole
        self.use_input        = use_input
        self.learn_init       = learn_init

        self.init             = initializations.get(init)
        self.inner_init       = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation       = activations.get(activation)
        self.relu             = activations.get('relu')
        self.inner_activation = activations.get(inner_activation)

        self.identity_connect = identity_connect
        self.axies            = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}  # only support at most 4D now!

        """
        Build the model weights.
        """
        # build the centroid grid (3 grid versions)
        self.grids = [Grid(output_dims,
                           input_dims,
                           -1,
                           peephole,
                           init, inner_init,
                           forget_bias_init,
                           activation, inner_activation, use_input,
                           name='Grid*{}'.format(k)
                           ) for k in xrange(3)]

        for k in xrange(3):
            self.grids[k].build()
            self._add(self.grids[k])

        # # input projection layer (projected to time-axis)       [x]
        # self.Ph  = Dense(input_dims[0], output_dims[0], name='Ph')
        # self.Pm  = Dense(input_dims[0], output_dims[0], name='Pm')
        #
        # self._add(self.Ph)
        # self._add(self.Pm)

        # learn init/
        if self.learn_init:
            self.hx0 = self.init((1, output_dims[0]))
            self.hy0 = self.init((1, output_dims[1]))
            self.mx0 = self.init((1, output_dims[0]))
            self.my0 = self.init((1, output_dims[1]))

            self.hx0.name, self.hy0.name = 'hx0', 'hy0'
            self.mx0.name, self.my0.name = 'mx0', 'my0'
            self.params += [self.hx0, self.hy0, self.mx0, self.my0]

        """
        Others info.
        """
        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def _step(self, *args):
        inputs = args[0]
        hx_tm1 = args[1]
        mx_tm1 = args[2]
        hy_tm1 = args[3]
        my_tm1 = args[4]

        # zero constant inputs.
        pre_info    = [[[T.zeros_like(hx_tm1)
                         for _ in xrange(self.depth)]
                         for _ in xrange(self.depth)]
                         for _ in xrange(4)]  # hx, mx, hy, my

        pre_inputs  = [[T.zeros_like(inputs)
                       for _ in xrange(self.depth)]
                       for _ in xrange(self.depth)]

        for kk in xrange(self.depth):
            pre_inputs[kk][kk] = inputs

        pre_info[0][0][0] = hx_tm1
        pre_info[1][0][0] = mx_tm1
        pre_info[2][0][0] = hy_tm1
        pre_info[3][0][0] = my_tm1

        for step_x in xrange(self.depth):
            for step_y in xrange(self.depth):
                # input hidden/memory/input information
                print pre_info[0][-1][-1], pre_info[2][-1][-1]

                hs_i  = [pre_info[0][step_x][step_y],
                         pre_info[2][step_x][step_y]]
                ms_i  = [pre_info[1][step_x][step_y],
                         pre_info[3][step_x][step_y]]
                xs_i  = [pre_inputs[step_x][step_y],
                         pre_inputs[step_x][step_y]]

                # compute grid-lstm
                if (step_x + step_y + 1) < self.depth:
                    hs_o, ms_o = self.grids[0].grid_(hs_i, ms_i, xs_i, priority =-1)
                elif (step_x + step_y + 1) == self.depth:
                    hs_o, ms_o = self.grids[1].grid_(hs_i, ms_i, xs_i, priority =-1)
                else:
                    hs_o, ms_o = self.grids[2].grid_(hs_i, ms_i, xs_i, priority =-1)

                # output hidden/memory information
                if (step_x == self.depth - 1) and (step_y == self.depth - 1):
                    hx_t, mx_t, hy_t, my_t = hs_o[0], ms_o[0], hs_o[1], ms_o[1]
                    return hx_t, mx_t, hy_t, my_t

                if step_x + 1 < self.depth:
                    pre_info[0][step_x + 1][step_y] = hs_o[0]
                    pre_info[1][step_x + 1][step_y] = ms_o[0]

                if step_y + 1 < self.depth:
                    pre_info[2][step_x][step_y + 1] = hs_o[1]
                    pre_info[3][step_x][step_y + 1] = ms_o[1]

    def __call__(self, X, init_x=None, init_y=None,
                 return_sequence=False, one_step=False):
        # recently we did not support masking.
        if X.ndim == 2:
            X = X[:, None, :]

        # one step
        if one_step:
            assert init_x is not None, 'previous x must be provided!'
            assert init_y is not None, 'previous y must be provided!'

        X = X.dimshuffle((1, 0, 2))
        if init_x is None:
            if self.learn_init:
                init_mx    = T.repeat(self.mx0, X.shape[1], axis=0)
                init_my    = T.repeat(self.my0, X.shape[1], axis=0)
                init_hx    = T.repeat(self.hx0, X.shape[1], axis=0)
                init_hy    = T.repeat(self.hy0, X.shape[1], axis=0)

                init_input = [init_hx, init_mx, init_hy, init_my]
            else:
                init_x     = [T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dims[0]), 1)] * 2
                init_y     = [T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dims[1]), 1)] * 2

                init_input = init_x + init_y
        else:
            init_input = init_x + init_y

        if not one_step:
            sequence       = [X]
            output_info    = init_input
            outputs, _     = theano.scan(
                self._step,
                sequences=sequence,
                outputs_info=output_info
            )
        else:
            outputs        = self._step(*([X[0]] + init_x + init_y))

        if return_sequence:
            hxs = outputs[0].dimshuffle((1, 0, 2))
            hys = outputs[2].dimshuffle((1, 0, 2))
            hs  = T.concatenate([hxs, hys], axis=-1)
            return hs
        else:
            hx  = outputs[0][-1]
            hy  = outputs[2][-1]
            h   = T.concatenate([hx, hy], axis=-1)
            return h