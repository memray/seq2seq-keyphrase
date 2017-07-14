__author__ = 'jiataogu'
import theano
# theano.config.exception_verbosity = 'high'
import logging

import emolga.basic.objectives as objectives
import emolga.basic.optimizers as optimizers
from emolga.layers.recurrent import *
from emolga.layers.embeddings import *
from emolga.models.encdec import RNNLM, Encoder, Decoder
from emolga.models.sandbox import SkipDecoder


logger = logging
RNN = JZS3  # change it here for other RNN models.
# Decoder = SkipDecoder


class VAE(RNNLM):
    """
    Variational Auto-Encoder: RNN-Variational Encoder/Decoder,
    in order to model the sentence generation.

    We implement the original VAE and a better version, IWAE.
    References:
        Auto-Encoding Variational Bayes
            http://arxiv.org/abs/1312.6114

        Importance Weighted Autoencoders
            http://arxiv.org/abs/1509.00519
    """

    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name   = 'vae'
        self.tparams= dict()

    def _add_tag(self, layer, tag):
        if tag not in self.tparams:
            self.tparams[tag] = []

        if layer:
            self.tparams[tag] += layer.params

    def build_(self):
        logger.info("build the variational auto-encoder")
        self.encoder = Encoder(self.config, self.rng, prefix='enc')
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', embed=self.encoder.Embed)
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec')

        # additional parameters for building Gaussian:
        logger.info("create Gaussian layers.")

        """
        Build the Gaussian distribution.
        """
        self.action_activ = activations.get('tanh')
        self.context_mean = Dense(
            self.config['enc_hidden_dim'] * 2
            if self.config['bidirectional']
            else self.config['enc_hidden_dim'],

            self.config['action_dim'],
            activation='linear',
            name="weight_mean"
        )

        self.context_std = Dense(
            self.config['enc_hidden_dim'] * 2
            if self.config['bidirectional']
            else self.config['enc_hidden_dim'],

            self.config['action_dim'],
            activation='linear',
            name="weight_std"
        )

        self.context_trans = Dense(
            self.config['action_dim'],
            self.config['dec_contxt_dim'],
            activation='tanh',
            name="transform"
        )

        # registration:
        self._add(self.context_mean)
        self._add(self.context_std)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # Q-layers:
        self._add_tag(self.encoder, 'q')
        self._add_tag(self.context_mean, 'q')
        self._add_tag(self.context_std, 'q')

        # P-layers:
        self._add_tag(self.decoder, 'p')
        self._add_tag(self.context_trans, 'p')

        # objectives and optimizers
        self.optimizer = optimizers.get(self.config['optimizer'])

        logger.info("create variational RECURRENT auto-encoder. ok")

    def compile_train(self):
        """
        build the training function here <:::>
        """
        # questions (theano variables)
        inputs = T.imatrix()  # padded input word sequence (for training)

        # encoding. (use backward encoding.)
        encoded = self.encoder.build_encoder(inputs[:, ::-1])

        # gaussian distribution
        mean = self.context_mean(encoded)
        ln_var = self.context_std(encoded)

        # [important] use multiple samples.
        if self.config['repeats'] > 1:
            L  = self.config['repeats']

            # repeat mean, ln_var and targets.
            func_r = lambda x: T.extra_ops.repeat(
                                x[:, None, :], L,
                                axis=1).reshape((x.shape[0] * L, x.shape[1]))
            mean, ln_var, target \
                   = [func_r(x) for x in [mean, ln_var, inputs]]
        else:
            target = inputs

        action  = mean + T.exp(ln_var / 2.) * self.rng.normal(mean.shape)
        context = self.context_trans(action)

        # decoding.
        logPxz, logPPL = self.decoder.build_decoder(target, context)

        # loss function for variational auto-encoding
        # regulation loss + reconstruction loss
        loss_reg = T.mean(objectives.get('GKL')(mean, ln_var))
        loss_rec = T.mean(-logPxz)
        loss_ppl = T.exp(T.mean(-logPPL))

        m_mean = T.mean(abs(mean))
        m_ln_var = T.mean(abs(ln_var))
        L1       = T.sum([T.sum(abs(w)) for w in self.params])

        loss = loss_reg + loss_rec
        updates = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs]

        self.train_ = theano.function(train_inputs,
                                      [loss_reg, loss_rec, L1, m_ln_var],
                                      updates=updates,
                                      name='train_fun')
        # add monitoring:
        self.monitor['action'] = action
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)
        logger.info("pre-training functions compile done.")

    def compile_sample(self):
        """
        build the sampler function here <:::>
        """
        # context vectors (as)
        self.decoder.build_sampler()

        l = T.iscalar()
        logger.info("compiling the computational graph :: action sampler")
        self.action_sampler = theano.function([l], self.rng.normal((l, self.config['action_dim'])))

        action = T.matrix()
        logger.info("compiling the compuational graph ::transform function::")
        self.transform = theano.function([action], self.context_trans(action))
        logger.info("display functions compile done.")

    def compile_inference(self):
        """
        build the hidden action prediction.
        """
        inputs = T.imatrix()  # padded input word sequence (for training)

        # encoding. (use backward encoding.)
        encoded = self.encoder.build_encoder(inputs[:, ::-1])

        # gaussian distribution
        mean    = self.context_mean(encoded)
        ln_var  = self.context_std(encoded)

        self.inference_ = theano.function([inputs], [encoded, mean, T.sqrt(T.exp(ln_var))])
        logger.info("inference function compile done.")

    def default_context(self):
        return self.transform(self.action_sampler(1))


class Helmholtz(VAE):
    """
    Another alternative I can think about is the Helmholtz Machine
    It is trained using a Reweighted Wake Sleep Algorithm.
    Reference:
        Reweighted Wake-Sleep
            http://arxiv.org/abs/1406.2751
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode = 'Evaluation',
                 dynamic_prior=False,
                 ):
        super(VAE, self).__init__(config, n_rng, rng)

        # self.config = config
        # self.n_rng = n_rng  # numpy random stream
        # self.rng = rng  # Theano random stream
        self.mode = mode
        self.name = 'multitask_helmholtz'
        self.tparams = dict()
        self.dynamic_prior = dynamic_prior

    def build_(self):
        logger.info('Build Helmholtz Recurrent Neural Networks')
        self.encoder = Encoder(self.config, self.rng, prefix='enc')
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', embed=self.encoder.Embed,
                                   highway=self.config['highway'])
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   highway=self.config['highway'])

        # The main difference between VAE and HM is that we can use
        # a more flexible prior instead of Gaussian here.
        # for example, we use a sigmoid prior here.

        """
        Build the Sigmoid Layers
        """
        # prior distribution (bias layer)
        self.Prior    = Constant(
            self.config['action_dim'],
            self.config['action_dim'],
            activation='sigmoid',
            name='prior_proj'
        )

        # Fake Posterior (Q-function)
        self.Posterior = Dense(
            self.config['enc_hidden_dim'] * 2
            if self.config['bidirectional']
            else self.config['enc_hidden_dim'],

            self.config['action_dim'],
            activation='sigmoid',
            name = 'posterior_proj'
        )

        # Action transform to context
        self.context_trans = Dense(
            self.config['action_dim'],
            self.config['dec_contxt_dim'],
            activation='linear',
            name="transform"
        )

        # registration:
        self._add(self.Posterior)
        self._add(self.Prior)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # Q-layers:
        self._add_tag(self.encoder, 'q')
        self._add_tag(self.Posterior, 'q')

        # P-layers:
        self._add_tag(self.Prior, 'p')
        self._add_tag(self.decoder, 'p')
        self._add_tag(self.context_trans, 'p')

        # objectives and optimizers
        self.optimizer_p = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})
        self.optimizer_q = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})

        logger.info("create Helmholtz RECURRENT neural network. ok")

    def dynamic(self):
        self.Prior   = Dense(
            self.config['state_dim'],
            self.config['action_dim'],
            activation='sigmoid',
            name='prior_proj'
        )

        self.params = []
        self.layers = []
        self.tparams= dict()

        # add layers again!
        # registration:
        self._add(self.Posterior)
        self._add(self.Prior)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # Q-layers:
        self._add_tag(self.encoder, 'q')
        self._add_tag(self.Posterior, 'q')

        # P-layers:
        self._add_tag(self.Prior, 'p')
        self._add_tag(self.decoder, 'p')
        self._add_tag(self.context_trans, 'p')

    def compile_(self, mode='train', contrastive=False):
        # compile the computational graph.
        # INFO: the parameters.
        # mode: 'train'/ 'display'/ 'policy' / 'all'

        ps = 'params: {\n'
        for p in self.params:
            ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        ps += '}.'
        logger.info(ps)

        param_num = np.sum([np.prod(p.shape.eval()) for p in self.params])
        logger.info("total number of the parameters of the model: {}".format(param_num))

        if mode == 'train' or mode == 'all':
            if not contrastive:
                self.compile_train()
            else:
                self.compile_train_CE()

        if mode == 'display' or mode == 'all':
            self.compile_sample()

        if mode == 'inference' or mode == 'all':
            self.compile_inference()

    def compile_train(self):
        """
        build the training function here <:::>
        """
        # get input sentence (x)
        inputs  = T.imatrix()  # padded input word sequence (for training)
        batch_size = inputs.shape[0]

        """
        The Computational Flow.
        """
        # encoding. (use backward encoding.)
        encoded = self.encoder.build_encoder(inputs[:, ::-1])

        # get Q(a|y) = sigmoid(.|Posterior * encoded)
        q_dis   = self.Posterior(encoded)

        # use multiple samples
        L  = T.iscalar('repeats') #self.config['repeats']

        def func_r(x):
            return T.extra_ops.repeat(x[:, None, :], L, axis=1).reshape((-1, x.shape[1]))  # ?

        q_dis, target = [func_r(x) for x in [q_dis, inputs]]

        # sample actions
        u       = self.rng.uniform(q_dis.shape)
        action  = T.cast(u <= q_dis, dtype=theano.config.floatX)

        # compute the exact probability for actions
        logQax  = T.sum(action * T.log(q_dis) + (1 - action) * T.log(1 - q_dis), axis=1)

        # decoding.
        context = self.context_trans(action)
        logPxa, count = self.decoder.build_decoder(target, context, return_count=True)
        logPPL  = logPxa / count
        # logPxa, logPPL = self.decoder.build_decoder(target, context)

        # prior.
        p_dis   = self.Prior(action)
        logPa   = T.sum(action * T.log(p_dis) + (1 - action) * T.log(1 - p_dis), axis=1)

        """
        Compute the weights
        """
        # reshape
        logQax  = logQax.reshape((batch_size, L))
        logPa   = logPa.reshape((batch_size, L))
        logPxa  = logPxa.reshape((batch_size, L))
        count   = count.reshape((batch_size, L))[:, :1]

        # P(x, a) = P(a) * P(x|a)
        logPx_a = logPa + logPxa
        log_wk  = logPx_a - logQax
        log_bpk = logPa - logQax

        log_w_sum  = logSumExp(log_wk, axis=1)
        log_bp_sum = logSumExp(log_bpk, axis=1)

        log_wnk    = log_wk - log_w_sum
        log_bpnk   = log_bpk - log_bp_sum

        # unbiased log-likelihood estimator
        # nll   = -T.mean(log_w_sum - T.log(L))
        nll        = T.mean(-(log_w_sum - T.log(L)))
        perplexity = T.exp(T.mean(-(log_w_sum - T.log(L)) / count))

        # perplexity = T.exp(-T.mean((log_w_sum - T.log(L)) / count))

        """
        Compute the Loss function
        """
        # loss    = weights * log [p(a)p(x|a)/q(a|x)]
        weights = T.exp(log_wnk)
        bp      = T.exp(log_bpnk)
        bq      = 1. / L
        ess     = T.mean(1 / T.sum(weights ** 2, axis=1))

        # monitoring
        # self.monitor['action'] = action
        if self.config['variant_control']:
            lossQ   = -T.mean(T.sum(logQax * (weights - bq), axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(logPa  * (weights - bp), axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))          # log p(x|a)
            lossP   = lossPxa + lossPa

            updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights, bp])
            updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weights])
        else:
            lossQ   = -T.mean(T.sum(logQax * weights, axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(logPa  * weights, axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))   # log p(x|a)
            lossP   = lossPxa + lossPa
            # lossRes = -T.mean(T.nnet.relu(T.sum((logPa + logPxa - logPx0) * weights, axis=1)))
            # lossP   = 0.1 * lossRes + lossP

            updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights])
            updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weights])

        updates   = updates_p + updates_q

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs] + [theano.Param(L, default=10)]

        self.train_ = theano.function(train_inputs,
                                      [lossPa, lossPxa, lossQ, perplexity, nll],
                                      updates=updates,
                                      name='train_fun')

        logger.info("compile the computational graph:: >__< :: explore function")
        self.explore_ = theano.function(train_inputs,
                                        [log_wk, count],
                                        name='explore_fun')

        # add monitoring:
        # self._monitoring()

        # compiling monitoring
        # self.compile_monitoring(train_inputs)
        logger.info("pre-training functions compile done.")

    def build_dynamics(self, states, action, Y):
        # this funtion is used to compute probabilities for language generation.
        # compute the probability of action
        assert self.dynamic_prior, 'only supports dynamic prior'
        p_dis     = self.Prior(states)
        logPa     = T.sum(action * T.log(p_dis) + (1 - action) * T.log(1 - p_dis), axis=1)
        context   = self.context_trans(action)
        logPxa, count = self.decoder.build_decoder(Y, context, return_count=True)
        return logPa, logPxa, count

    def compile_sample(self):
        """
        build the sampler function here <:::>
        """
        # context vectors (as)
        self.decoder.build_sampler()

        logger.info("compiling the computational graph :: action sampler")
        if self.dynamic_prior:
            states = T.matrix()
            p_dis  = self.Prior(states)
            u      = self.rng.uniform(p_dis.shape)
        else:
            p_dis  = self.Prior()
            l      = T.iscalar()
            u      = self.rng.uniform((l, p_dis.shape[-1]))

        action  = T.cast(u <= p_dis, dtype=theano.config.floatX)

        if self.dynamic_prior:
            self.action_sampler = theano.function([states], action)
        else:
            self.action_sampler = theano.function([l], action)

        # compute the action probability
        logPa   = T.sum(action * T.log(p_dis) + (1 - action) * T.log(1 - p_dis), axis=1)
        if self.dynamic_prior:
            self.action_prob = theano.function([states, action], logPa)
        else:
            self.action_prob = theano.function([action], logPa)

        action  = T.matrix()
        logger.info("compiling the computational graph ::transform function::")
        self.transform = theano.function([action], self.context_trans(action))
        logger.info("display functions compile done.")

    def compile_inference(self):
        """
        build the hidden action prediction.
        """
        inputs = T.imatrix()  # padded input word sequence (for training)

        # encoding. (use backward encoding.)
        encoded = self.encoder.build_encoder(inputs[:, ::-1])

        # get Q(a|y) = sigmoid(.|Posterior * encoded)
        q_dis   = self.Posterior(encoded)
        p_dis   = self.Prior(inputs)

        self.inference_ = theano.function([inputs], [encoded, q_dis, p_dis])
        logger.info("inference function compile done.")

    def evaluate_(self, inputs):
        """
        build the evaluation function for valid/testing
        Note that we need multiple sampling for this!
        """
        log_wks = []
        count   = None
        N       = self.config['eval_N']
        L       = self.config['eval_repeats']

        for _ in xrange(N):
            log_wk, count = self.explore_(inputs, L)
            log_wks.append(log_wk)

        log_wk     = np.concatenate(log_wks, axis=1)
        log_wk_sum = logSumExp(log_wk, axis=1, status='numpy')

        nll        = np.mean(-(log_wk_sum - np.log(N * L)))
        perplexity = np.exp(np.mean(-(log_wk_sum - np.log(N * L)) / count))

        return nll, perplexity

    """
    OLD CODE::  >>> It doesn't work !
    """
    def compile_train_CE(self):
        # compile the computation graph (use contrastive noise, for 1 sample here. )

        """
        build the training function here <:::>
        """
        # get input sentence (x)
        inputs  = T.imatrix()  # padded input word sequence x (for training)
        noises  = T.imatrix()  # padded noise word sequence y (it stands for another question.)
        batch_size = inputs.shape[0]

        """
        The Computational Flow.
        """
        # encoding. (use backward encoding.)
        encodex = self.encoder.build_encoder(inputs[:, ::-1])
        encodey = self.encoder.build_encoder(noises[:, ::-1])

        # get Q(a|y) = sigmoid(.|Posterior * encoded)
        q_dis_x = self.Posterior(encodex)
        q_dis_y = self.Posterior(encodey)

        # use multiple samples
        if self.config['repeats'] > 1:
            L  = self.config['repeats']

            # repeat mean, ln_var and targets.
            func_r = lambda x: T.extra_ops.repeat(
                                x[:, None, :], L,
                                axis=1).reshape((x.shape[0] * L, x.shape[1]))
            q_dis_x, q_dis_y, target \
                   = [func_r(x) for x in [q_dis_x, q_dis_y, inputs]]
        else:
            target = inputs
            L  = 1

        # sample actions
        u = self.rng.uniform(q_dis_x.shape)
        action  = T.cast(u <= q_dis_x, dtype=theano.config.floatX)

        # compute the exact probability for actions (for data distribution)
        logQax  = T.sum(action * T.log(q_dis_x) + (1 - action) * T.log(1 - q_dis_x), axis=1)

        # compute the exact probability for actions (for noise distribution)
        logQay  = T.sum(action * T.log(q_dis_y) + (1 - action) * T.log(1 - q_dis_y), axis=1)

        # decoding.
        context = self.context_trans(action)
        logPxa, count = self.decoder.build_decoder(target, context, return_count=True)

        # prior.
        p_dis   = self.Prior(target)
        logPa   = T.sum(action * T.log(p_dis) + (1 - action) * T.log(1 - p_dis), axis=1)

        """
        Compute the weights
        """
        # reshape
        logQax  = logQax.reshape((batch_size, L))
        logQay  = logQay.reshape((batch_size, L))
        logPa   = logPa.reshape((batch_size, L))
        logPxa  = logPxa.reshape((batch_size, L))

        # P(x, a) = P(a) * P(x|a)
        # logPx_a = logPa + logPxa
        logPx_a    = logPa + logPxa

        # normalizing the weights
        log_wk     = logPx_a - logQax
        log_bpk    = logPa - logQax

        log_w_sum  = logSumExp(log_wk, axis=1)
        log_bp_sum = logSumExp(log_bpk, axis=1)

        log_wnk    = log_wk - log_w_sum
        log_bpnk   = log_bpk - log_bp_sum

        # unbiased log-likelihood estimator
        logPx   = T.mean(log_w_sum - T.log(L))
        perplexity = T.exp(-T.mean((log_w_sum - T.log(L)) / count))

        """
        Compute the Loss function
        """
        # loss    = weights * log [p(a)p(x|a)/q(a|x)]
        weights = T.exp(log_wnk)
        bp      = T.exp(log_bpnk)
        bq      = 1. / L
        ess     = T.mean(1 / T.sum(weights ** 2, axis=1))

        """
        Contrastive Estimation
        """
        # lossQ   = -T.mean(T.sum(logQax * (weights - bq), axis=1))   # log q(a|x)
        logC    = logQax - logQay
        weightC = weights * (1 - T.nnet.sigmoid(logC))
        lossQ   = -T.mean(T.sum(logC * weightC, axis=1))
        # lossQT  = -T.mean(T.sum(T.log(T.nnet.sigmoid(logC)) * weights, axis=1))

        # monitoring
        self.monitor['action'] = logC

        """
        Maximum-likelihood Estimation
        """
        lossPa  = -T.mean(T.sum(logPa  * (weights - bp), axis=1))   # log p(a)
        lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))          # log p(x|a)
        lossP   = lossPxa + lossPa
        # loss    = lossQT + lossPa + lossPxa

        updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights, bp])
        updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weightC])
        updates   = updates_p + updates_q

        logger.info("compiling the compuational graph ::training function::")
        train_inputs  = [inputs, noises]

        self.train_ce_ = theano.function(train_inputs,
                                        [lossPa, lossPxa, lossQ, perplexity, ess],
                                        updates=updates,
                                        name='train_fun')

        # add monitoring:
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)

        logger.info("pre-training functions compile done.")


class HarX(Helmholtz):
    """
    Another alternative I can think about is the Helmholtz Machine
    It is trained using a Reweighted Wake Sleep Algorithm.
    Reference:
        Reweighted Wake-Sleep
            http://arxiv.org/abs/1406.2751

    We extend the original Helmholtz Machine to a recurrent way.
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode = 'Evaluation',
                 dynamic_prior=False,
                 ):
        super(VAE, self).__init__(config, n_rng, rng)

        # self.config = config
        # self.n_rng = n_rng  # numpy random stream
        # self.rng = rng  # Theano random stream
        self.mode = mode
        self.name = 'multitask_helmholtz'
        self.tparams = dict()
        self.dynamic_prior = dynamic_prior

    def build_(self):
        logger.info('Build Helmholtz Recurrent Neural Networks')

        # backward encoder
        self.encoder = Encoder(self.config, self.rng, prefix='enc')

        # feedforward + hidden content decoder
        self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   embed=self.encoder.Embed
                                   if self.config['shared_embed']
                                   else None)

        # The main difference between VAE and HM is that we can use
        # a more flexible prior instead of Gaussian here.
        # for example, we use a sigmoid prior here.

        """
        Build the Sigmoid Layers
        """
        # prior distribution (conditional distribution)
        self.Prior    = Dense(
            self.config['dec_hidden_dim'],
            self.config['action_dim'],
            activation='sigmoid',
            name='prior_proj'
        )

        # Fake Posterior (Q-function)
        if self.config['decposterior']:
            self.Posterior = Dense2(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['dec_hidden_dim'],
                self.config['action_dim'],
                activation='sigmoid',
                name='posterior_proj'
            )
        else:
            self.Posterior = Dense(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['action_dim'],
                activation='sigmoid',
                name='posterior_proj'
            )

        # Action transform to context
        self.context_trans = Dense(
            self.config['action_dim'],
            self.config['dec_contxt_dim'],
            activation='linear',
            name="transform"
        )

        # registration:
        self._add(self.Posterior)
        self._add(self.Prior)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # Q-layers:
        self._add_tag(self.encoder, 'q')
        self._add_tag(self.Posterior, 'q')

        # P-layers:
        self._add_tag(self.Prior, 'p')
        self._add_tag(self.decoder, 'p')
        self._add_tag(self.context_trans, 'p')

        # objectives and optimizers
        self.optimizer_p = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})
        self.optimizer_q = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})

        logger.info("create Helmholtz RECURRENT neural network. ok")

    def compile_(self, mode='train', contrastive=False):
        # compile the computational graph.
        # INFO: the parameters.
        # mode: 'train'/ 'display'/ 'policy' / 'all'

        ps = 'params: {\n'
        for p in self.params:
            ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        ps += '}.'
        logger.info(ps)

        param_num = np.sum([np.prod(p.shape.eval()) for p in self.params])
        logger.info("total number of the parameters of the model: {}".format(param_num))

        if mode == 'train' or mode == 'all':
            self.compile_train()

        if mode == 'display' or mode == 'all':
            self.compile_sample()

        if mode == 'inference' or mode == 'all':
            self.compile_inference()

    """
    Training
    """
    def compile_train(self):
        """
        build the training function here <:::>
        """
        # get input sentence (x)
        inputs     = T.imatrix()  # padded input word sequence (for training)
        batch_size = inputs.shape[0]

        logger.info(
            """
            The Computational Flow. ---> In a recurrent fashion

            [= v =] <:::
            Inference-Generation in one scan

            >>>> Encoding without hidden variable. (use backward encoding.)
            """
        )
        embeded, mask \
                   = self.decoder.Embed(inputs, True)  # (nb_samples, max_len, embedding_dim)
        encoded    = self.encoder.build_encoder(inputs[:, ::-1], return_sequence=True)[:, ::-1, :]
        count      = T.cast(T.sum(mask, axis=1), dtype=theano.config.floatX)[:, None]  # (nb_samples,)

        logger.info(
            """
            >>>> Repeat
            """
        )
        L          = T.iscalar('repeats')              # self.config['repeats']

        def _repeat(x, dimshuffle=True):
            if x.ndim == 3:
                y = T.extra_ops.repeat(x[:, None, :, :], L, axis=1).reshape((-1, x.shape[1], x.shape[2]))
                if dimshuffle:
                    y = y.dimshuffle(1, 0, 2)
            else:
                y = T.extra_ops.repeat(x[:, None, :], L, axis=1).reshape((-1, x.shape[1]))
                if dimshuffle:
                    y = y.dimshuffle(1, 0)
            return y

        embeded    = _repeat(embeded)                  # (max_len, nb_samples * L, embedding_dim)
        encoded    = _repeat(encoded)                  # (max_len, nb_samples * L, enc_hidden_dim)
        target     = _repeat(inputs, False)            # (nb_samples * L, max_len)
        mask       = _repeat(mask, False)              # (nb_samples * L, max_len)
        init_dec   = T.zeros((encoded.shape[1],
                              self.config['dec_hidden_dim']),
                              dtype='float32')      # zero initialization
        uniform    = self.rng.uniform((embeded.shape[0],
                                       embeded.shape[1],
                                       self.config['action_dim'])) # uniform dirstribution pre-sampled.

        logger.info(
            """
            >>>> Recurrence
            """
        )

        def _recurrence(embed_t, enc_t, u_t, dec_tm1):
            """
            x_t:   (nb_samples, dec_embedd_dim)
            enc_t: (nb_samples, enc_hidden_dim)
            dec_t: (nb_samples, dec_hidden_dim)
            """
            # get q(z_t|dec_t, enc_t);  sample z_t; compute the Posterior (inference) prob.
            if self.config['decposterior']:
                q_dis_t   = self.Posterior(enc_t, dec_tm1)
            else:
                q_dis_t   = self.Posterior(enc_t)

            z_t       = T.cast(u_t <= q_dis_t, dtype='float32')
            log_qzx_t = T.sum(z_t * T.log(q_dis_t) + (1 - z_t) * T.log(1 - q_dis_t), axis=1)  # (nb_samples * L, )

            # compute the prior probability
            p_dis_t   = self.Prior(dec_tm1)
            log_pz0_t = T.sum(z_t * T.log(p_dis_t) + (1 - z_t) * T.log(1 - p_dis_t), axis=1)

            # compute the decoding probability
            context_t = self.context_trans(z_t)
            readout_t = self.decoder.hidden_readout(dec_tm1) + self.decoder.context_readout(context_t)
            for l in self.decoder.output_nonlinear:
                readout_t = l(readout_t)
            pxz_dis_t = self.decoder.output(readout_t)

            # compute recurrence
            dec_t   = self.decoder.RNN(embed_t, C=context_t, init_h=dec_tm1, one_step=True)

            return dec_t, z_t, log_qzx_t, log_pz0_t, pxz_dis_t

        # (max_len, nb_samples, ?)
        outputs, _ = theano.scan(
            _recurrence,
            sequences=[embeded, encoded, uniform],
            outputs_info=[init_dec, None, None, None, None])
        _, z, log_qzx, log_pz0, pxz_dis = outputs

        # summary of scan/ dimshuffle/ reshape
        def _grab_prob(probs, x):
            assert probs.ndim == 3
            b_size     = probs.shape[0]
            max_len    = probs.shape[1]
            vocab_size = probs.shape[2]
            probs      = probs.reshape((b_size * max_len, vocab_size))
            return probs[T.arange(b_size * max_len), x.flatten(1)].reshape(x.shape)  # advanced indexing

        log_qzx    = T.sum(log_qzx.dimshuffle(1, 0) * mask, axis=-1).reshape((batch_size, L))
        log_pz0    = T.sum(log_pz0.dimshuffle(1, 0) * mask, axis=-1).reshape((batch_size, L))
        log_pxz    = T.sum(T.log(_grab_prob(pxz_dis.dimshuffle(1, 0, 2), target)) * mask, axis=-1).reshape((batch_size, L))

        logger.info(
            """
            >>>> Compute the weights [+ _ =]
            """
        )
        log_pxnz   = log_pz0  + log_pxz    # log p(X, Z)
        log_wk     = log_pxnz - log_qzx    # log[p(X, Z)/q(Z|X)]
        log_bpk    = log_pz0  - log_qzx    # log[p(Z)/q(Z|X)]

        log_w_sum  = logSumExp(log_wk, axis=1)
        log_bp_sum = logSumExp(log_bpk, axis=1)

        log_wnk    = log_wk - log_w_sum
        log_bpnk   = log_bpk - log_bp_sum

        # unbiased log-likelihood estimator [+ _ =]
        # Finally come to this place
        nll        = T.mean(-(log_w_sum - T.log(L)))
        perplexity = T.exp(T.mean(-(log_w_sum - T.log(L)) / count))

        # perplexity = T.exp(-T.mean((log_w_sum - T.log(L)) / count))

        logger.info(
            """
            >>>> Compute the gradients [+ _ =]
            """
        )
        # loss    = weights * log [p(a)p(x|a)/q(a|x)]
        weights = T.exp(log_wnk)
        bp      = T.exp(log_bpnk)
        bq      = 1. / L
        ess     = T.mean(1 / T.sum(weights ** 2, axis=1))

        # monitoring
        self.monitor['hidden state'] = z
        if self.config['variant_control']:
            lossQ   = -T.mean(T.sum(log_qzx * (weights - bq), axis=1))   # log q(z|x)
            lossPa  = -T.mean(T.sum(log_pz0 * (weights - bp), axis=1))   # log p(z)
            lossPxa = -T.mean(T.sum(log_pxz * weights, axis=1))          # log p(x|z)
            lossP   = lossPxa + lossPa

            # L2 regu
            lossP  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['p']])
            lossQ  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['q']])

            updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights, bp])
            updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weights])
        else:
            lossQ   = -T.mean(T.sum(log_qzx * weights, axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(log_pz0 * weights, axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(log_pxz * weights, axis=1))   # log p(x|a)
            lossP   = lossPxa + lossPa

            # L2 regu
            print 'L2 ?'
            lossP  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['p']])
            lossQ  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['q']])

            updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights])
            updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weights])

        updates   = updates_p + updates_q
        logger.info("compiling the compuational graph:: >__< ::training function::")
        train_inputs = [inputs] + [theano.Param(L, default=10)]

        self.train_ = theano.function(train_inputs,
                                      [lossPa, lossPxa, lossQ, perplexity, nll],
                                      updates=updates,
                                      name='train_fun')

        logger.info("compile the computational graph:: >__< :: explore function")
        self.explore_ = theano.function(train_inputs,
                                        [log_wk, count],
                                        name='explore_fun')

        # add monitoring:
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)
        logger.info("pre-training functions compile done.")

    def generate_(self, context=None, max_len=None, mode='display'):
        # overwrite the RNNLM generator as there are hidden variables every time step
        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'] if not max_len else max_len,
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None)


class THarX(Helmholtz):
    """
    Another alternative I can think about is the Helmholtz Machine
    It is trained using a Reweighted Wake Sleep Algorithm.
    Reference:
        Reweighted Wake-Sleep
            http://arxiv.org/abs/1406.2751

    We extend the original Helmholtz Machine to a recurrent way.
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode = 'Evaluation',
                 dynamic_prior=False,
                 ):
        super(VAE, self).__init__(config, n_rng, rng)

        # self.config = config
        # self.n_rng = n_rng  # numpy random stream
        # self.rng = rng  # Theano random stream
        self.mode = mode
        self.name = 'multitask_helmholtz'
        self.tparams = dict()
        self.dynamic_prior = dynamic_prior

    def build_(self):
        logger.info('Build Helmholtz Recurrent Neural Networks')

        # backward encoder
        self.encoder = Encoder(self.config, self.rng, prefix='enc')

        # feedforward + hidden content decoder
        self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   embed=self.encoder.Embed
                                   if self.config['shared_embed']
                                   else None)

        # The main difference between VAE and HM is that we can use
        # a more flexible prior instead of Gaussian here.
        # for example, we use a sigmoid prior here.

        """
        Build the Sigmoid Layers
        """
        # prior distribution (conditional distribution)
        self.Prior    = Dense(
            self.config['dec_hidden_dim'],
            self.config['action_dim'],
            activation='softmax',
            name='prior_proj'
        )

        # Fake Posterior (Q-function)
        if self.config['decposterior']:
            self.Posterior = Dense2(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['dec_hidden_dim'],
                self.config['action_dim'],
                activation='softmax',
                name='posterior_proj'
            )
        else:
            self.Posterior = Dense(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['action_dim'],
                activation='softmax',
                name='posterior_proj'
            )

        # Action transform to context
        self.context_trans = Dense(
            self.config['action_dim'],
            self.config['dec_contxt_dim'],
            activation='linear',
            name="transform"
        )

        # registration:
        self._add(self.Posterior)
        self._add(self.Prior)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # Q-layers:
        self._add_tag(self.encoder, 'q')
        self._add_tag(self.Posterior, 'q')

        # P-layers:
        self._add_tag(self.Prior, 'p')
        self._add_tag(self.decoder, 'p')
        self._add_tag(self.context_trans, 'p')

        # objectives and optimizers
        self.optimizer_p = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})
        self.optimizer_q = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})

        logger.info("create Helmholtz RECURRENT neural network. ok")

    def compile_(self, mode='train', contrastive=False):
        # compile the computational graph.
        # INFO: the parameters.
        # mode: 'train'/ 'display'/ 'policy' / 'all'

        ps = 'params: {\n'
        for p in self.params:
            ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        ps += '}.'
        logger.info(ps)

        param_num = np.sum([np.prod(p.shape.eval()) for p in self.params])
        logger.info("total number of the parameters of the model: {}".format(param_num))

        if mode == 'train' or mode == 'all':
            self.compile_train()

        if mode == 'display' or mode == 'all':
            self.compile_sample()

        if mode == 'inference' or mode == 'all':
            self.compile_inference()

    """
    Training
    """
    def compile_train(self):
        """
        build the training function here <:::>
        """
        # get input sentence (x)
        inputs     = T.imatrix('inputs')  # padded input word sequence (for training)
        batch_size = inputs.shape[0]

        logger.info(
            """
            The Computational Flow. ---> In a recurrent fashion

            [= v =] <:::
            Inference-Generation in one scan

            >>>> Encoding without hidden variable. (use backward encoding.)
            """
        )
        embeded, mask \
                   = self.decoder.Embed(inputs, True)  # (nb_samples, max_len, embedding_dim)
        encoded    = self.encoder.build_encoder(inputs[:, ::-1], return_sequence=True)[:, ::-1, :]
        count      = T.cast(T.sum(mask, axis=1), dtype=theano.config.floatX)[:, None]  # (nb_samples,)

        logger.info(
            """
            >>>> Repeat
            """
        )
        L          = T.iscalar('repeats')              # self.config['repeats']

        def _repeat(x, dimshuffle=True):
            if x.ndim == 3:
                y = T.extra_ops.repeat(x[:, None, :, :], L, axis=1).reshape((-1, x.shape[1], x.shape[2]))
                if dimshuffle:
                    y = y.dimshuffle(1, 0, 2)
            else:
                y = T.extra_ops.repeat(x[:, None, :], L, axis=1).reshape((-1, x.shape[1]))
                if dimshuffle:
                    y = y.dimshuffle(1, 0)
            return y

        embeded    = _repeat(embeded)                  # (max_len, nb_samples * L, embedding_dim)
        encoded    = _repeat(encoded)                  # (max_len, nb_samples * L, enc_hidden_dim)
        target     = _repeat(inputs, False)            # (nb_samples * L, max_len)
        mask       = _repeat(mask, False)              # (nb_samples * L, max_len)
        init_dec   = T.zeros((encoded.shape[1],
                              self.config['dec_hidden_dim']),
                              dtype='float32')      # zero initialization
        # uniform    = self.rng.uniform((embeded.shape[0],
        #                                embeded.shape[1],
        #                                self.config['action_dim'])) # uniform dirstribution pre-sampled.

        logger.info(
            """
            >>>> Recurrence
            """
        )

        def _recurrence(embed_t, enc_t, dec_tm1):
            """
            x_t:   (nb_samples, dec_embedd_dim)
            enc_t: (nb_samples, enc_hidden_dim)
            dec_t: (nb_samples, dec_hidden_dim)
            """
            # get q(z_t|dec_t, enc_t);  sample z_t; compute the Posterior (inference) prob.
            if self.config['decposterior']:
                q_dis_t   = self.Posterior(enc_t, dec_tm1)
            else:
                q_dis_t   = self.Posterior(enc_t)

            z_t       = self.rng.multinomial(pvals=q_dis_t, dtype='float32')
            log_qzx_t = T.sum(T.log(q_dis_t) * z_t, axis=1)
            # log_qzx_t = T.log(q_dis_t[T.arange(q_dis_t.shape[0]), z_t])

            # z_t       = T.cast(u_t <= q_dis_t, dtype='float32')
            # log_qzx_t = T.sum(z_t * T.log(q_dis_t) + (1 - z_t) * T.log(1 - q_dis_t), axis=1)  # (nb_samples * L, )

            # compute the prior probability
            p_dis_t   = self.Prior(dec_tm1)
            log_pz0_t = T.sum(T.log(p_dis_t) * z_t, axis=1)
            # log_pz0_t = T.log(p_dis_t[T.arange(p_dis_t.shape[0]), z_t])
            # log_pz0_t = T.sum(z_t * T.log(p_dis_t) + (1 - z_t) * T.log(1 - p_dis_t), axis=1)

            # compute the decoding probability
            context_t = self.context_trans(z_t)
            readout_t = self.decoder.hidden_readout(dec_tm1) + self.decoder.context_readout(context_t)
            for l in self.decoder.output_nonlinear:
                readout_t = l(readout_t)
            pxz_dis_t = self.decoder.output(readout_t)

            # compute recurrence
            dec_t   = self.decoder.RNN(embed_t, C=context_t, init_h=dec_tm1, one_step=True)

            return dec_t, z_t, log_qzx_t, log_pz0_t, pxz_dis_t

        # (max_len, nb_samples, ?)
        outputs, scan_update = theano.scan(
            _recurrence,
            sequences=[embeded, encoded],
            outputs_info=[init_dec, None, None, None, None])
        _, z, log_qzx, log_pz0, pxz_dis = outputs

        # summary of scan/ dimshuffle/ reshape
        def _grab_prob(probs, x):
            assert probs.ndim == 3
            b_size     = probs.shape[0]
            max_len    = probs.shape[1]
            vocab_size = probs.shape[2]
            probs      = probs.reshape((b_size * max_len, vocab_size))
            return probs[T.arange(b_size * max_len), x.flatten(1)].reshape(x.shape)  # advanced indexing

        log_qzx    = T.sum(log_qzx.dimshuffle(1, 0) * mask, axis=-1).reshape((batch_size, L))
        log_pz0    = T.sum(log_pz0.dimshuffle(1, 0) * mask, axis=-1).reshape((batch_size, L))
        log_pxz    = T.sum(T.log(_grab_prob(pxz_dis.dimshuffle(1, 0, 2), target)) * mask, axis=-1).reshape((batch_size, L))

        logger.info(
            """
            >>>> Compute the weights [+ _ =]
            """
        )
        log_pxnz   = log_pz0  + log_pxz    # log p(X, Z)
        log_wk     = log_pxnz - log_qzx    # log[p(X, Z)/q(Z|X)]
        log_bpk    = log_pz0  - log_qzx    # log[p(Z)/q(Z|X)]

        log_w_sum  = logSumExp(log_wk, axis=1)
        log_bp_sum = logSumExp(log_bpk, axis=1)

        log_wnk    = log_wk - log_w_sum
        log_bpnk   = log_bpk - log_bp_sum

        # unbiased log-likelihood estimator [+ _ =]
        # Finally come to this place
        nll        = T.mean(-(log_w_sum - T.log(L)))
        perplexity = T.exp(T.mean(-(log_w_sum - T.log(L)) / count))

        # perplexity = T.exp(-T.mean((log_w_sum - T.log(L)) / count))

        logger.info(
            """
            >>>> Compute the gradients [+ _ =]
            """
        )
        # loss    = weights * log [p(a)p(x|a)/q(a|x)]
        weights = T.exp(log_wnk)
        bp      = T.exp(log_bpnk)
        bq      = 1. / L
        ess     = T.mean(1 / T.sum(weights ** 2, axis=1))

        # monitoring
        self.monitor['hidden state'] = z
        if self.config['variant_control']:
            lossQ   = -T.mean(T.sum(log_qzx * (weights - bq), axis=1))   # log q(z|x)
            lossPa  = -T.mean(T.sum(log_pz0 * (weights - bp), axis=1))   # log p(z)
            lossPxa = -T.mean(T.sum(log_pxz * weights, axis=1))          # log p(x|z)
            lossP   = lossPxa + lossPa

            # L2 regu
            lossP  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['p']])
            lossQ  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['q']])

            updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights, bp])
            updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weights])
        else:
            lossQ   = -T.mean(T.sum(log_qzx * weights, axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(log_pz0 * weights, axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(log_pxz * weights, axis=1))   # log p(x|a)
            lossP   = lossPxa + lossPa

            # L2 regu
            print 'L2 ?'
            lossP  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['p']])
            lossQ  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['q']])

            updates_p = self.optimizer_p.get_updates(self.tparams['p'], [lossP, weights])
            updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, weights])

        updates   = updates_p + updates_q + scan_update
        logger.info("compiling the compuational graph:: >__< ::training function::")
        train_inputs = [inputs] + [theano.Param(L, default=10)]

        self.train_ = theano.function(train_inputs,
                                      [lossPa, lossPxa, lossQ, perplexity, nll],
                                      updates=updates,
                                      name='train_fun')

        logger.info("compile the computational graph:: >__< :: explore function")
        self.explore_ = theano.function(train_inputs,
                                        [log_wk, count],
                                        updates=scan_update,
                                        name='explore_fun')

        # add monitoring:
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs, updates=scan_update)
        logger.info("pre-training functions compile done.")

    def generate_(self, context=None, max_len=None, mode='display'):
        # overwrite the RNNLM generator as there are hidden variables every time step
        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'] if not max_len else max_len,
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None)


class NVTM(Helmholtz):
    """
    Neural Variational Topical Models
    We use the Neural Variational Inference and Learning (NVIL) to build the
    learning, instead of using Helmholtz Machine(Reweighted Wake-sleep)
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode = 'Evaluation',
                 dynamic_prior=False,
                 ):
        super(VAE, self).__init__(config, n_rng, rng)

        self.mode = mode
        self.name = 'neural_variational'
        self.tparams = dict()
        self.dynamic_prior = dynamic_prior

    def build_(self):
        logger.info('Build Helmholtz Recurrent Neural Networks')

        # backward encoder
        self.encoder = Encoder(self.config, self.rng, prefix='enc')

        # feedforward + hidden content decoder
        self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   embed=self.encoder.Embed
                                   if self.config['shared_embed']
                                   else None)

        # The main difference between VAE and NVIL is that we can use
        # a more flexible prior instead of Gaussian here.
        # for example, we use a softmax prior here.

        """
        Build the Prior Layer (Conditional Prior)
        """
        # prior distribution (conditional distribution)
        self.Prior    = Dense(
            self.config['dec_hidden_dim'],
            self.config['action_dim'],
            activation='softmax',
            name='prior_proj'
        )

        if self.config['decposterior']:   # we use both enc/dec net as input.

            # Variational Posterior (Q-function)
            self.Posterior = Dense2(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['dec_hidden_dim'],
                self.config['action_dim'],
                activation='softmax',
                name='posterior_proj'
            )

            # Baseline Estimator
            self.C_lambda1 = Dense2(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['dec_hidden_dim'],
                100,
                activation='tanh',
                name='baseline-1')
            self.C_lambda2 = Dense(100, 1, activation='linear',
                                   name='baseline-2')
        else:

            # Variational Posterior
            self.Posterior = Dense(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                self.config['action_dim'],
                activation='softmax',
                name='posterior_proj'
            )

            # Baseline Estimator
            self.C_lambda1 = Dense(
                self.config['enc_hidden_dim']
                        if not self.config['bidirectional']
                        else 2 * self.config['enc_hidden_dim'],
                100,
                activation='tanh',
                name='baseline-1')
            self.C_lambda2 = Dense(100, 1, activation='linear',
                                   name='baseline-2')

        # Action transform to context
        self.context_trans = Dense(
            self.config['action_dim'],
            self.config['dec_contxt_dim'],
            activation='linear',
            name="transform"
        )

        # registration:
        self._add(self.Posterior)
        self._add(self.Prior)
        self._add(self.context_trans)
        self._add(self.C_lambda1)
        self._add(self.C_lambda2)

        self._add(self.encoder)
        self._add(self.decoder)

        # Q-layers:
        self._add_tag(self.encoder, 'q')
        self._add_tag(self.Posterior, 'q')

        # P-layers:
        self._add_tag(self.Prior, 'p')
        self._add_tag(self.decoder, 'p')
        self._add_tag(self.context_trans, 'p')

        # Lambda-layers
        self._add_tag(self.C_lambda1, 'l')
        self._add_tag(self.C_lambda2, 'l')

        # c/v
        self.c = shared_scalar(0., dtype='float32')
        self.v = shared_scalar(1., dtype='float32')

        # objectives and optimizers
        self.optimizer_p = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})
        self.optimizer_q = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})
        self.optimizer_l = optimizers.get(self.config['optimizer'], kwargs={'clipnorm': 5})

        logger.info("create Neural Variational Topic Network. ok")

    def compile_(self, mode='train', contrastive=False):
        # compile the computational graph.
        # INFO: the parameters.
        # mode: 'train'/ 'display'/ 'policy' / 'all'

        ps = 'params: {\n'
        for p in self.params:
            ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        ps += '}.'
        logger.info(ps)

        param_num = np.sum([np.prod(p.shape.eval()) for p in self.params])
        logger.info("total number of the parameters of the model: {}".format(param_num))

        if mode == 'train' or mode == 'all':
            self.compile_train()

        if mode == 'display' or mode == 'all':
            self.compile_sample()

        if mode == 'inference' or mode == 'all':
            self.compile_inference()

    """
    Training
    """
    def compile_train(self):
        """
        build the training function here <:::>
        """
        # get input sentence (x)
        inputs     = T.imatrix('inputs')  # padded input word sequence (for training)
        batch_size = inputs.shape[0]

        logger.info(
            """
            The Computational Flow. ---> In a recurrent fashion

            [= v =] <:::
            Inference-Generation in one scan

            >>>> Encoding without hidden variable. (use backward encoding.)
            """
        )
        embeded, mask \
                   = self.decoder.Embed(inputs, True)  # (nb_samples, max_len, embedding_dim)
        mask       = T.cast(mask, dtype='float32')

        encoded    = self.encoder.build_encoder(inputs[:, ::-1], return_sequence=True)[:, ::-1, :]

        L          = T.iscalar('repeats')              # self.config['repeats']

        def _repeat(x, dimshuffle=True):
            if x.ndim == 3:
                y = T.extra_ops.repeat(x[:, None, :, :], L, axis=1).reshape((-1, x.shape[1], x.shape[2]))
                if dimshuffle:
                    y = y.dimshuffle(1, 0, 2)
            else:
                y = T.extra_ops.repeat(x[:, None, :], L, axis=1).reshape((-1, x.shape[1]))
                if dimshuffle:
                    y = y.dimshuffle(1, 0)
            return y

        embeded    = _repeat(embeded)                  # (max_len, nb_samples * L, embedding_dim)
        encoded    = _repeat(encoded)                  # (max_len, nb_samples * L, enc_hidden_dim)
        target     = _repeat(inputs, False)            # (nb_samples * L, max_len)
        mask       = _repeat(mask, False)
        count      = T.cast(T.sum(mask, axis=1), dtype=theano.config.floatX)[:, None]  # (nb_samples,)

        init_dec   = T.zeros((encoded.shape[1],
                              self.config['dec_hidden_dim']),
                              dtype='float32')         # zero initialization

        logger.info(
            """
            >>>> Recurrence
            """
        )

        def _recurrence(embed_t, enc_t, dec_tm1):
            """
            x_t:   (nb_samples, dec_embedd_dim)
            enc_t: (nb_samples, enc_hidden_dim)
            dec_t: (nb_samples, dec_hidden_dim)
            """
            # get q(z_t|dec_t, enc_t);  sample z_t;
            # compute the Posterior (inference) prob.
            # compute the baseline estimator
            if self.config['decposterior']:
                q_dis_t   = self.Posterior(enc_t, dec_tm1)
                c_lmd_t   = self.C_lambda2(self.C_lambda1(enc_t, dec_tm1)).flatten(1)

            else:
                q_dis_t   = self.Posterior(enc_t)
                c_lmd_t   = self.C_lambda2(self.C_lambda1(enc_t)).flatten(1)

            # sampling
            z_t       = self.rng.multinomial(pvals=q_dis_t, dtype='float32')
            log_qzx_t = T.sum(T.log(q_dis_t) * z_t, axis=1)

            # compute the prior probability
            p_dis_t   = self.Prior(dec_tm1)
            log_pz0_t = T.sum(T.log(p_dis_t) * z_t, axis=1)

            # compute the decoding probability
            context_t = self.context_trans(z_t)
            readout_t = self.decoder.hidden_readout(dec_tm1) + self.decoder.context_readout(context_t)
            for l in self.decoder.output_nonlinear:
                readout_t = l(readout_t)
            pxz_dis_t = self.decoder.output(readout_t)

            # compute recurrence
            dec_t   = self.decoder.RNN(embed_t, C=context_t, init_h=dec_tm1, one_step=True)

            return dec_t, z_t, log_qzx_t, log_pz0_t, pxz_dis_t, c_lmd_t

        # (max_len, nb_samples, ?)
        outputs, scan_update = theano.scan(
            _recurrence,
            sequences=[embeded, encoded],
            outputs_info=[init_dec, None, None, None, None, None])
        _, z, log_qzx, log_pz0, pxz_dis, c_lmd = outputs

        # summary of scan/ dimshuffle/ reshape
        def _grab_prob(probs, x):
            assert probs.ndim == 3
            b_size     = probs.shape[0]
            max_len    = probs.shape[1]
            vocab_size = probs.shape[2]
            probs      = probs.reshape((b_size * max_len, vocab_size))
            return probs[T.arange(b_size * max_len), x.flatten(1)].reshape(x.shape)  # advanced indexing

        logger.info(
            """
            >>>> Compute the weights [+ _ =]
            """
        )
        # log Q/P and C
        log_qzx    = log_qzx.dimshuffle(1, 0) * mask
        log_pz0    = log_pz0.dimshuffle(1, 0) * mask
        log_pxz    = T.log(_grab_prob(pxz_dis.dimshuffle(1, 0, 2), target)) * mask
        c_lambda   = c_lmd.dimshuffle(1, 0) * mask

        Lb         = T.sum(log_pz0 + log_pxz - log_qzx, axis=-1)   # lower bound
        l_lambda   = log_pz0 + log_pxz - log_qzx - c_lambda

        alpha      = T.cast(0.0, dtype='float32')
        numel      = T.sum(mask)

        cb         = T.sum(l_lambda) / numel
        vb         = T.sum(l_lambda ** 2) / T.sum(mask) - cb ** 2
        c          = self.c * alpha + (1 - alpha) * cb  # T.cast(cb, dtype='float32')
        v          = self.v * alpha + (1 - alpha) * vb  # T.cast(vb, dtype='float32')

        l_normal   = (l_lambda - c) / T.max((1., T.sqrt(v))) * mask
        l_base     = T.mean(T.sum(l_normal, axis=1))
        nll        = T.mean(-Lb)                 # variational lower-bound
        perplexity = T.exp(T.mean(-Lb[:, None] / count))  # perplexity of lower-bound

        logger.info(
            """
            >>>> Compute the gradients [+ _ =]
            """
        )

        # monitoring
        self.monitor['hidden state'] = z

        lossP   = -T.mean(T.sum(log_pxz  + log_pz0,  axis=1))
        lossQ   = -T.mean(T.sum(log_qzx  * l_normal, axis=1))
        lossL   = -T.mean(T.sum(c_lambda * l_normal, axis=1))   # ||L - c - c_lambda||2-> 0

        # lossP   = -T.sum(log_pxz  + log_pz0)  / numel
        # lossQ   = -T.sum(log_qzx  * l_normal) / numel
        # lossL   = -T.sum(c_lambda * l_normal) / numel  # ||L - c - c_lambda||2-> 0
        #
        # # L2 regu
        # print 'L2 ?'
        # lossP  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['p']])
        # lossQ  += 0.0001 * T.sum([T.sum(p**2) for p in self.tparams['q']])

        updates_p = self.optimizer_p.get_updates(self.tparams['p'], lossP)
        updates_q = self.optimizer_q.get_updates(self.tparams['q'], [lossQ, l_normal])
        updates_l = self.optimizer_l.get_updates(self.tparams['l'], [lossL, l_normal])

        updates   = updates_p + updates_q + updates_l + scan_update
        updates  += [(self.c, c), (self.v, v)]

        logger.info("compiling the compuational graph:: >__< ::training function::")
        train_inputs = [inputs] + [theano.Param(L, default=1)]

        self.train_ = theano.function(train_inputs,
                                      [lossL, lossP, lossQ, perplexity, nll, l_base],
                                      updates=updates,
                                      name='train_fun')

        logger.info("compile the computational graph:: >__< :: explore function")
        self.explore_ = theano.function(train_inputs,
                                        [lossL, lossP, lossQ, perplexity, nll, l_base],
                                        updates=scan_update,
                                        name='explore_fun')

        # add monitoring:
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs, updates=scan_update)
        logger.info("pre-training functions compile done.")

    def generate_(self, context=None, max_len=None, mode='display'):
        # overwrite the RNNLM generator as there are hidden variables every time step
        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'] if not max_len else max_len,
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None)
