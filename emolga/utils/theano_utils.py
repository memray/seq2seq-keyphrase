from __future__ import absolute_import

from theano import gof
from theano.tensor import basic as tensor
import numpy as np
import theano
import theano.tensor as T


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val), name=name)


def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)


def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](0.), *dims)


def alloc_ones_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](1.), *dims)


def ndim_tensor(ndim):
    if ndim == 1:
        return T.vector()
    elif ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()


# get int32 tensor
def ndim_itensor(ndim, name=None):
    if ndim == 2:
        return T.imatrix(name)
    elif ndim == 3:
        return T.itensor3(name)
    elif ndim == 4:
        return T.itensor4(name)
    return T.imatrix(name)


# dot-product
def dot(inp, matrix, bias=None):
    """
    Decide the right type of dot product depending on the input
    arguments
    """
    if 'int' in inp.dtype and inp.ndim == 2:
        return matrix[inp.flatten()]
    elif 'int' in inp.dtype:
        return matrix[inp]
    elif 'float' in inp.dtype and inp.ndim == 3:
        shape0 = inp.shape[0]
        shape1 = inp.shape[1]
        shape2 = inp.shape[2]
        if bias:
            return (T.dot(inp.reshape((shape0 * shape1, shape2)), matrix) + bias).reshape((shape0, shape1, matrix.shape[1]))
        else:
            return T.dot(inp.reshape((shape0 * shape1, shape2)), matrix).reshape((shape0, shape1, matrix.shape[1]))
    else:
        if bias:
            return T.dot(inp, matrix) + bias
        else:
            return T.dot(inp, matrix)


# Numerically stable log(sum(exp(A))). Can also be used in softmax function.
def logSumExp(x, axis=None, mask=None, status='theano', c=None, err=1e-7):
    """
        Numerically stable log(sum(exp(A))). Can also be used in softmax function.
        c is the additional input when it doesn't require masking but x need.

    """
    if status == 'theano':
        J = T
    else:
        J = np

    if c is None:
        x_max = J.max(x, axis=axis, keepdims=True)
    else:
        x_max = J.max(J.concatenate([c, x], axis=-1), axis=axis, keepdims=True)

    if c is None:
        if not mask:
            l_t = J.sum(J.exp(x - x_max), axis=axis, keepdims=True)

        else:
            l_t = J.sum(J.exp(x - x_max) * mask, axis=axis, keepdims=True)
    else:
        if not mask:
            l_t = J.sum(J.exp(x - x_max), axis=axis, keepdims=True) + \
                  J.sum(J.exp(c - x_max), axis=axis, keepdims=True)
        else:
            l_t = J.sum(J.exp(x - x_max) * mask, axis=axis, keepdims=True) + \
                  J.sum(J.exp(c - x_max), axis=axis, keepdims=True)

    x_t = J.log(J.maximum(l_t, err)) + x_max
    return x_t


def softmax(x):
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def masked_softmax(x, mask, err=1e-9):
    assert x.ndim == 2, 'support two-dimension'
    weights  = softmax(x)
    weights *= mask
    weights  = weights / (T.sum(weights, axis=-1)[:, None] + err) * mask
    return weights


def cosine_sim(k, M):
    k_unit = k / (T.sqrt(T.sum(k**2)) + 1e-5)
    # T.patternbroadcast(k_unit.reshape((1,k_unit.shape[0])),(True,False))
    k_unit = k_unit.dimshuffle(('x', 0))
    k_unit.name = "k_unit"
    M_lengths = T.sqrt(T.sum(M**2, axis=1)).dimshuffle((0, 'x'))
    M_unit = M / (M_lengths + 1e-5)
    M_unit.name = "M_unit"
    return T.sum(k_unit * M_unit, axis=1)


def cosine_sim2d(k, M):
    # k: (nb_samples, memory_width)
    # M: (nb_samples, memory_dim, memory_width)

    # norms of keys and memories
    k_norm = T.sqrt(T.sum(T.sqr(k), 1)) + 1e-5  # (nb_samples,)
    M_norm = T.sqrt(T.sum(T.sqr(M), 2)) + 1e-5  # (nb_samples, memory_dim,)

    k      = k[:, None, :]                      # (nb_samples, 1, memory_width)
    k_norm = k_norm[:, None]                    # (nb_samples, 1)

    sim    = T.sum(k * M, axis=2)               # (nb_samples, memory_dim,)
    sim   /= k_norm * M_norm                    # (nb_samples, memory_dim,)
    return sim


def dot_2d(k, M, b=None, g=None):
    # k: (nb_samples, memory_width)
    # M: (nb_samples, memory_dim, memory_width)

    # norms of keys and memories
    # k_norm = T.sqrt(T.sum(T.sqr(k), 1)) + 1e-5  # (nb_samples,)
    # M_norm = T.sqrt(T.sum(T.sqr(M), 2)) + 1e-5  # (nb_samples, memory_dim,)

    k      = k[:, None, :]                      # (nb_samples, 1, memory_width)
    value  = k * M
    if b is not None:
        b  = b[:, None, :]
        value *= b         # (nb_samples, memory_dim,)

    if g is not None:
        g  = g[None, None, :]
        value *= g

    sim    = T.sum(value, axis=2)
    return sim


def shift_convolve(weight, shift, shift_conv):
    shift = shift.dimshuffle((0, 'x'))
    return T.sum(shift * weight[shift_conv], axis=0)


def shift_convolve2d(weight, shift, shift_conv):
    return T.sum(shift[:, :, None] * weight[:, shift_conv], axis=1)
