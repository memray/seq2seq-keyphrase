import theano.tensor as T


def softmax(x):
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def vector_softmax(x):
    return T.nnet.softmax(x.reshape((1, x.shape[0])))[0]


def time_distributed_softmax(x):
    import warnings
    warnings.warn("time_distributed_softmax is deprecated. Just use softmax!", DeprecationWarning)
    return softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def relu(x):
    return T.nnet.relu(x)


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work
    '''
    return x


def maxout2(x):
    shape = x.shape
    if x.ndim == 1:
        shape1 = T.cast(shape[0] / 2, 'int64')
        shape2 = T.cast(2, 'int64')
        x = x.reshape([shape1, shape2])
        x = x.max(1)
    elif x.ndim == 2:
        shape1 = T.cast(shape[1] / 2, 'int64')
        shape2 = T.cast(2, 'int64')
        x = x.reshape([shape[0], shape1, shape2])
        x = x.max(2)
    elif x.ndim == 3:
        shape1 = T.cast(shape[2] / 2, 'int64')
        shape2 = T.cast(2, 'int64')
        x = x.reshape([shape[0], shape[1], shape1, shape2])
        x = x.max(3)
    return x


from emolga.utils.generic_utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
