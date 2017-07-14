__author__ = 'jiataogu'
from .core import *
"""
Attention Model.
    <::: Two kinds of attention models ::::>
    -- Linear Transformation
    -- Inner Product
"""


class Attention(Layer):
    def __init__(self, target_dim, source_dim, hidden_dim,
                 init='glorot_uniform', name='attention',
                 coverage=False, max_len=50,
                 shared=False):

        super(Attention, self).__init__()
        self.init       = initializations.get(init)
        self.softmax    = activations.get('softmax')
        self.tanh       = activations.get('tanh')
        self.target_dim = target_dim
        self.source_dim = source_dim
        self.hidden_dim = hidden_dim
        self.max_len    = max_len
        self.coverage   = coverage

        if coverage:
            print('Use Coverage Trick!')

        self.Wa         = self.init((self.target_dim, self.hidden_dim))
        self.Ua         = self.init((self.source_dim, self.hidden_dim))
        self.va         = self.init((self.hidden_dim, 1))

        self.Wa.name, self.Ua.name, self.va.name = \
                '{}_Wa'.format(name), '{}_Ua'.format(name), '{}_va'.format(name)
        self.params     = [self.Wa, self.Ua, self.va]
        if coverage:
            self.Ca      = self.init((1, self.hidden_dim))
            self.Ca.name = '{}_Ca'.format(name)
            self.params += [self.Ca]

    def __call__(self, X, S,
                 Smask=None,
                 return_log=False,
                 Cov=None):
        assert X.ndim + 1 == S.ndim, 'source should be one more dimension than target.'
        # X is the decoder representation of t-1:    (nb_samples, hidden_dims)
        # S is the hidden representation of source text:    (nb_samples, maxlen_s, context_dim)
        # X_mask: mask, an array showing which elements in X are not 0 [nb_sample, max_len]
        # Cov is the coverage vector (nb_samples, maxlen_s)

        if X.ndim == 1:
            X = X[None, :]
            S = S[None, :, :]
            if not Smask:
                Smask = Smask[None, :]

        Eng   = dot(X[:, None, :], self.Wa) + dot(S, self.Ua)  # (nb_samples, source_num, hidden_dims)
        Eng   = self.tanh(Eng)
        # location aware:
        if self.coverage:
            Eng += dot(Cov[:, :, None], self.Ca)  # (nb_samples, source_num, hidden_dims)

        Eng   = dot(Eng, self.va)
        Eng   = Eng[:, :, 0]                      # ? (nb_samples, source_num)

        if Smask is not None:
            # I want to use mask!
            EngSum = logSumExp(Eng, axis=1, mask=Smask)
            if return_log:
                return (Eng - EngSum) * Smask
            else:
                return T.exp(Eng - EngSum) * Smask
        else:
            if return_log:
                return T.log(self.softmax(Eng))
            else:
                return self.softmax(Eng)


class CosineAttention(Layer):
    def __init__(self, target_dim, source_dim,
                 init='glorot_uniform',
                 use_pipe=True,
                 name='attention'):

        super(CosineAttention, self).__init__()
        self.init       = initializations.get(init)
        self.softmax    = activations.get('softmax')
        self.softplus   = activations.get('softplus')
        self.tanh       = activations.get('tanh')
        self.use_pipe   = use_pipe

        self.target_dim = target_dim
        self.source_dim = source_dim

        # pipe
        if self.use_pipe:
            self.W_key  = Dense(self.target_dim, self.source_dim, name='W_key')
        else:
            assert target_dim == source_dim
            self.W_key  = Identity(name='W_key')
        self._add(self.W_key)

        # sharpen
        # self.W_beta     = Dense(self.target_dim, 1, name='W_beta')
        # dio-sharpen
        # self.W_beta     = Dense(self.target_dim, self.source_dim, name='W_beta')
        # self._add(self.W_beta)

        # self.gamma      = self.init((source_dim, ))
        # self.gamma      = self.init((target_dim, source_dim))
        # self.gamma.name = 'o_gamma'
        # self.params    += [self.gamma]

    def __call__(self, X, S, Smask=None, return_log=False):
        assert X.ndim + 1 == S.ndim, 'source should be one more dimension than target.'

        if X.ndim == 1:
            X = X[None, :]
            S = S[None, :, :]
            if not Smask:
                Smask = Smask[None, :]

        key   = self.W_key(X)                   # (nb_samples, source_dim)
        # beta  = self.softplus(self.W_beta(X))   # (nb_samples, source_dim)

        Eng   = dot_2d(key, S)  #, g=self.gamma)
        # Eng   = cosine_sim2d(key, S)  # (nb_samples, source_num)
        # Eng   = T.repeat(beta, Eng.shape[1], axis=1) * Eng

        if Smask is not None:
            # I want to use mask!
            EngSum = logSumExp(Eng, axis=1, mask=Smask)
            if return_log:
                return (Eng - EngSum) * Smask
            else:
                return T.exp(Eng - EngSum) * Smask
        else:
            if return_log:
                return T.log(self.softmax(Eng))
            else:
                return self.softmax(Eng)

