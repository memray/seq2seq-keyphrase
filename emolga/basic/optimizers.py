from __future__ import absolute_import
import theano
import sys

from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T
import logging

from emolga.utils.theano_utils import shared_zeros, shared_scalar, floatX
from emolga.utils.generic_utils import get_from_module
from six.moves import zip
from copy import copy, deepcopy

logger = logging.getLogger(__name__)


def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * T.log(p / p_hat)


class Optimizer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates   = []
        self.save_parm = []

    def add(self, v):
        self.save_parm += [v]

    def get_state(self):
        return [u[0].get_value() for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            u[0].set_value(floatX(v))

    def get_updates(self, params, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        """
        Consider the situation that gradient is weighted.
        """
        if isinstance(loss, list):
            grads = T.grad(loss[0], params, consider_constant=loss[1:])  # gradient of loss
        else:
            grads = T.grad(loss, params)

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            print('use gradient clipping!!')
            print('clipnorm = %f' % self.clipnorm)
            norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        else:
            print('not use gradient clipping!!')

        return grads

    def get_config(self):
        return {"name": self.__class__.__name__}


class SGD(Optimizer):

    def __init__(self, lr=0.05, momentum=0.9, decay=0.01, nesterov=True, *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)
        self.lr = shared_scalar(lr)
        self.momentum = shared_scalar(momentum)

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g in zip(params, grads):
            m = shared_zeros(p.get_value().shape)  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, new_p))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "momentum": float(self.momentum.get_value()),
                "decay": float(self.decay.get_value()),
                "nesterov": self.nesterov}


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)
        self.rho = shared_scalar(rho)
        self.iterations = shared_scalar(0)

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2  # update accumulator
            self.updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, new_p))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "rho": float(self.rho.get_value()),
                "epsilon": self.epsilon}


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = a + g ** 2  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "epsilon": self.epsilon}


class Adadelta(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1212.5701
    '''
    def __init__(self, lr=0.1, rho=0.95, epsilon=1e-6, *args, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)
        self.iterations = shared_scalar(0)

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        delta_accumulators = [shared_zeros(p.get_value().shape) for p in params]
        # self.updates = []
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2  # update accumulator
            self.updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a +
                                                             self.epsilon)

            new_p = p - self.lr * update
            self.updates.append((p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * update ** 2
            self.updates.append((d_a, new_d_a))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "rho": self.rho,
                "epsilon": self.epsilon}


class Adam(Optimizer):  # new Adam is designed for our purpose.
    '''
        Reference: http://arxiv.org/abs/1412.6980v8

        Default parameters follow those provided in the original paper.
        We add Gaussian Noise to improve the performance.
    '''
    def __init__(self, lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, save=False, rng=None, *args, **kwargs):
        print('args=%s' % str(args))
        print('kwargs=%s' % str(kwargs))
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        print(locals())

        # if 'iterations' in kwargs:
        #     print('iterations=%s' % str(kwargs['iterations']))
        #     self.iterations = shared_scalar(kwargs['iterations'],  name='iteration')
        # else:
        #     print('iterations not set')
        #     self.iterations = shared_scalar(0,  name='iteration')
        self.iterations = shared_scalar(0, name='iteration')
        self.lr         = shared_scalar(lr, name='lr')
        # self.rng        = MRG_RandomStreams(use_cuda=True)
        self.noise      = []
        self.forget     = dict()
        # self.rng        = rng
        self.beta_1     = beta_1
        self.beta_2     = beta_2
        self.epsilon    = epsilon

        self.add(self.iterations)
        self.add(self.lr)

    def add_noise(self, param):
        if param.name not in self.noise:
            logger.info('add gradient noise to {}'.format(param))
            self.noise += [param.name]

    def add_forget(self, param):
        if param.name not in self.forget:
            logger.info('add forgetting list to {}'.format(param))
            self.forget[param.name] = theano.shared(param.get_value())

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1.)]
        self.pu = []

        t = self.iterations + 1
        lr_t = self.lr * T.sqrt(1 - self.beta_2**t) / (1 - self.beta_1**t)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0., name=p.name + '_m')  # zero init of moment
            v = theano.shared(p.get_value() * 0., name=p.name + '_v')  # zero init of velocity

            self.add(m)
            self.add(v)

            # g_noise = self.rng.normal(g.shape, 0, T.sqrt(0.005 * t ** (-0.55)), dtype='float32')

            # if p.name in self.noise:
            #     g_deviated = g + g_noise
            # else:
            #     g_deviated = g

            g_deviated = g  #  + g_noise
            m_t = (self.beta_1 * m) + (1 - self.beta_1) * g_deviated
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * (g_deviated**2)
            u_t = -lr_t * m_t / (T.sqrt(v_t) + self.epsilon)
            p_t = p + u_t

            # # memory reformatting!
            # if p.name in self.forget:
            #     p_t = (1 - p_mem) * p_t + p_mem * self.forget[p.name]
            #     p_s = (1 - p_fgt) * p_t + p_fgt * self.forget[p.name]
            #     self.updates.append((self.forget[p.name], p_s))

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            self.updates.append((p, p_t))  # apply constraints
            self.pu.append((p, p_t - p))

        if self.save:
            return self.updates, self.pu
        return self.updates

    def get_config(self):
        # print(theano.tensor.cast(self.lr, dtype='float32').eval())
        # print(int(theano.tensor.cast(self.iterations, dtype='int32').eval()))
        config = {'lr':     float(theano.tensor.cast(self.lr, dtype='float32').eval()),
                  'beta_1': float(self.beta_1),
                  'beta_2': float(self.beta_2),
                  'iterations':  int(theano.tensor.cast(self.iterations, dtype='int32').eval()),
                  'noise':  self.noise
                  }
        base_config = super(Adam, self).get_config()
        return_config = dict(list(base_config.items()) + list(config.items()))
        print('Getting config of optimizer: \n\t\t %s' % str(return_config))
        return return_config

# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True,
                           kwargs=kwargs)
