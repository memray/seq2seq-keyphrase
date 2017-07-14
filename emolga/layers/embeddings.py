# -*- coding: utf-8 -*-

from .core import Layer
from emolga.utils.theano_utils import *
import emolga.basic.initializations as initializations


class Embedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''

    def __init__(self, input_dim, output_dim, init='uniform', name=None):

        super(Embedding, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = self.init((self.input_dim, self.output_dim))

        self.params = [self.W]

        if name is not None:
            self.set_name(name)

    def get_output_mask(self, X):
        '''
        T.ones_like(X): Return an array of ones with shape and type of input.
            T.eq(X, 0): X==0?
            1 - T.eq(X, 0): X!=0?
        :return an array shows that which x!=0
        '''
        return T.ones_like(X) * (1 - T.eq(X, 0))

    def __call__(self, X, mask_zero=False, context=None):
        '''
        return the embedding of X
        :param X:         a set of words, all the X have same length due to padding
                            shape=[nb_sample, max_len]
        :param mask_zero: whether return the mask of X, a list of [0,1] showing which x!=0
        :param context:
        :return
                emb_X:    embedding of X, shape = [nb_sample, max_len, emb_dim]
                X_mask:   mask of X,      shape=[nb_sample, max_len]

        '''
        if context is None:
            out = self.W[X]
        else:
            assert context.ndim == 3
            flag  = False
            if X.ndim == 1:
                flag = True
                X = X[:, None]

            b_size = context.shape[0]

            EMB = T.repeat(self.W[None, :, :], b_size, axis=0)
            EMB = T.concatenate([EMB, context], axis=1)

            m_size = EMB.shape[1]
            e_size = EMB.shape[2]
            maxlen = X.shape[1]

            EMB = EMB.reshape((b_size * m_size, e_size))
            Z   = (T.arange(b_size)[:, None] * m_size + X).reshape((b_size * maxlen,))
            out = EMB[Z]  # (b_size * maxlen, e_size)

            if not flag:
                out = out.reshape((b_size, maxlen, e_size))
            else:
                out = out.reshape((b_size, e_size))

        if mask_zero:
            return out, T.cast(self.get_output_mask(X), dtype='float32')
        else:
            return out


class Zero(Layer):
    def __call__(self, X):
        out = T.zeros(X.shape)
        return out


class Bias(Layer):
    def __call__(self, X):
        tmp = X.flatten()
        tmp = tmp.dimshuffle(0, 'x')
        return tmp
