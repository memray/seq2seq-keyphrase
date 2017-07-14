# -*- coding: utf-8 -*-

from emolga.utils.theano_utils import *
import emolga.basic.initializations as initializations
import emolga.basic.activations as activations


class Layer(object):
    def __init__(self):
        self.params  = []
        self.layers  = []
        self.monitor = {}
        self.watchlist = []

    def init_updates(self):
        self.updates = []

    def _monitoring(self):
        # add monitoring variables
        for l in self.layers:
            for v in l.monitor:
                name = v + '@' + l.name
                print(name)
                self.monitor[name] = l.monitor[v]

    def __call__(self, X, *args, **kwargs):
        return X

    def _add(self, layer):
        if layer:
            self.layers.append(layer)
            self.params += layer.params

    def supports_masked_input(self):
        ''' Whether or not this layer respects the output mask of its previous layer in its calculations. If you try
        to attach a layer that does *not* support masked_input to a layer that gives a non-None output_mask() that is
        an error'''
        return False

    def get_output_mask(self, train=None):
        '''
        For some models (such as RNNs) you want a way of being able to mark some output data-points as
        "masked", so they are not used in future calculations. In such a model, get_output_mask() should return a mask
        of one less dimension than get_output() (so if get_output is (nb_samples, nb_timesteps, nb_dimensions), then the mask
        is (nb_samples, nb_timesteps), with a one for every unmasked datapoint, and a zero for every masked one.

        If there is *no* masking then it shall return None. For instance if you attach an Activation layer (they support masking)
        to a layer with an output_mask, then that Activation shall also have an output_mask. If you attach it to a layer with no
        such mask, then the Activation's get_output_mask shall return None.

        Some emolga have an output_mask even if their input is unmasked, notably Embedding which can turn the entry "0" into
        a mask.
        '''
        return None

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            if p.eval().shape != w.shape:
                raise Exception("Layer shape %s not compatible with weight shape %s." % (p.eval().shape, w.shape))
            p.set_value(floatX(w))

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights

    def get_params(self):
        return self.params

    def set_name(self, name):
        for i in range(len(self.params)):
            if self.params[i].name is None:
                self.params[i].name = '%s_p%d' % (name, i)
            else:
                self.params[i].name = name + '_' + self.params[i].name
        self.name = name


class MaskedLayer(Layer):
    '''
    If your layer trivially supports masking (by simply copying the input mask to the output), then subclass MaskedLayer
    instead of Layer, and make sure that you incorporate the input mask into your calculation of get_output()
    '''
    def supports_masked_input(self):
        return True


class Identity(Layer):
    def __init__(self, name='Identity'):
        super(Identity, self).__init__()
        if name is not None:
            self.set_name(name)

    def __call__(self, X):
        return X


class Dense(Layer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='tanh', name='Dense',
                 learn_bias=True, negative_bias=False):

        super(Dense, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = (activation == 'linear')

        # self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        if not negative_bias:
            self.b = shared_zeros((self.output_dim))
        else:
            self.b = shared_ones((self.output_dim))

        self.learn_bias = learn_bias
        if self.learn_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.b.name = '%s_b' % name

    def __call__(self, X):
        # output = self.activation(T.dot(X, self.W) + 4. * self.b) # why with a 4.0 here? change to 1
        output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def reverse(self, Y):
        assert self.linear

        output = T.dot((Y - self.b), self.W.T)
        return output


class Dense2(Layer):
    def __init__(self, input_dim1, input_dim2, output_dim, init='glorot_uniform', activation='tanh', name='Dense', learn_bias=True):

        super(Dense2, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.linear = (activation == 'linear')

        # self.input = T.matrix()

        self.W1 = self.init((self.input_dim1, self.output_dim))
        self.W2 = self.init((self.input_dim2, self.output_dim))
        self.b  = shared_zeros((self.output_dim))

        self.learn_bias = learn_bias
        if self.learn_bias:
            self.params = [self.W1, self.W2, self.b]
        else:
            self.params = [self.W1, self.W2]

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W1.name = '%s_W1' % name
        self.W2.name = '%s_W2' % name
        self.b.name = '%s_b' % name

    def __call__(self, X1, X2):
        output = self.activation(T.dot(X1, self.W1) + T.dot(X2, self.W2) + self.b)
        return output


class Constant(Layer):
    def __init__(self, input_dim, output_dim, init=None, activation='tanh', name='Bias'):

        super(Constant, self).__init__()
        assert input_dim == output_dim, 'Bias Layer needs to have the same input/output nodes.'

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.b = shared_zeros(self.output_dim)
        self.params = [self.b]

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.b.name = '%s_b' % name

    def __call__(self, X=None):
        output = self.activation(self.b)
        if X:
            L = X.shape[0]
            output = T.extra_ops.repeat(output[None, :], L, axis=0)
        return output


class MemoryLinear(Layer):
    def __init__(self, input_dim, input_wdth, init='glorot_uniform',
                 activation='tanh', name='Bias', has_input=True):
        super(MemoryLinear, self).__init__()

        self.init       = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim  = input_dim
        self.input_wdth = input_wdth

        self.b = self.init((self.input_dim, self.input_wdth))
        self.params = [self.b]

        if has_input:
            self.P = self.init((self.input_dim, self.input_wdth))
            self.params += [self.P]

        if name is not None:
            self.set_name(name)

    def __call__(self, X=None):
        out = self.b[None, :, :]
        if X:
            out += self.P[None, :, :] * X
        return self.activation(out)


class Dropout(MaskedLayer):
    """
        Hinton's dropout.
    """
    def __init__(self, rng=None, p=1., name=None):
        super(Dropout, self).__init__()
        self.p   = p
        self.rng = rng

    def __call__(self, X, train=True):
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= self.rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X


class Activation(MaskedLayer):
    """
        Apply an activation function to an output.
    """
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activations.get(activation)

    def __call__(self, X):
        return self.activation(X)

