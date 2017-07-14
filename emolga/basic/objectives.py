from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.


def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)).mean(axis=-1)


def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)


def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce


def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce


def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred + epsilon), axis=-1)

####################################################
# Variational Auto-encoder

def gaussian_kl_divergence(mean, ln_var):
    """Computes the KL-divergence of Gaussian variables from the standard one.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function returns a variable
    representing the KL-divergence between the given multi-dimensional Gaussian
    :math:`N(\\mu, S)` and the standard Gaussian :math:`N(0, I)`

    .. math::

       D_{\\mathbf{KL}}(N(\\mu, S) \\| N(0, I)),

    where :math:`S` is a diagonal matrix such that :math:`S_{ii} = \\sigma_i^2`
    and :math:`I` is an identity matrix.

    Args:
        mean (~chainer.Variable): A variable representing mean of given
            gaussian distribution, :math:`\\mu`.
        ln_var (~chainer.Variable): A variable representing logarithm of
            variance of given gaussian distribution, :math:`\\log(\\sigma^2)`.

    Returns:
        ~chainer.Variable: A variable representing KL-divergence between
            given gaussian distribution and the standard gaussian.

    """
    var = T.exp(ln_var)
    return  0.5 * T.sum(mean * mean + var - ln_var - 1, 1)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
gkl = GKL = gaussian_kl_divergence

from emolga.utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
