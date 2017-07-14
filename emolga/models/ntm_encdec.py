__author__ = 'jiataogu'

import theano
theano.config.exception_verbosity = 'high'

import logging
import copy

import emolga.basic.objectives as objectives
import emolga.basic.optimizers as optimizers
from emolga.layers.recurrent import *
from emolga.layers.ntm_minibatch import Controller, BernoulliController
from emolga.layers.embeddings import *
from core import Model

logger = logging.getLogger(__name__)
RNN    = JZS3              # change it here for other RNN models.


class RecurrentBase(Model):
    """
    The recurrent base for SimpleRNN, GRU, JZS3, LSTM and Neural Turing Machines
    """
    def __init__(self, config, model='RNN', prefix='enc', use_contxt=True, name=None):
        super(RecurrentBase, self).__init__()

        self.config     = config
        self.model      = model
        self.prefix     = prefix
        self.use_contxt = use_contxt
        if not name:
            self.name   = self.prefix
        else:
            self.name   = name

        if self.config['binary']:
            NTM         = BernoulliController
        else:
            NTM         = Controller

        def _build_RNN():
            logger.info('BUILD::>>>>>>>> Gated Recurrent Units.')
            core = RNN(
                self.config['{}_embedd_dim'.format(self.prefix)],
                self.config['{}_hidden_dim'.format(self.prefix)],
                self.config['{}_contxt_dim'.format(self.prefix)] if use_contxt else None,
                name='{}_rnn'.format(self.prefix)
            )

            if self.config['bias_code']:
                init = Dense(
                    self.config['{}_contxt_dim'.format(self.prefix)],
                    self.config['{}_hidden_dim'.format(self.prefix)],
                    activation='tanh',
                    name='{}_init'.format(self.prefix)
                )
            else:
                init = Zero()

            return core, [init]

        def _build_NTM():
            """
            Build a simple Neural Turing Machine.
            We use a feedforward controller here.
            """
            logger.info('BUILD::>>>>>>>> Controller Units.')
            core = NTM(
                self.config['{}_embedd_dim'.format(self.prefix)],
                self.config['{}_memory_dim'.format(self.prefix)],
                self.config['{}_memory_wdth'.format(self.prefix)],
                self.config['{}_hidden_dim'.format(self.prefix)],
                self.config['{}_shift_width'.format(self.prefix)],
                name="{}_ntm".format(self.prefix),
                readonly=self.config['{}_read-only'.format(self.prefix)],
                curr_input=self.config['{}_curr_input'.format(self.prefix)],
                recurrence=self.config['{}_recurrence'.format(self.prefix)]
            )

            if self.config['bias_code']:
                raise NotImplementedError
            else:
                init_w = T.nnet.softmax(initializations.get('glorot_uniform')((1, self.config['{}_memory_dim'.format(self.prefix)])))
                init_r = T.nnet.softmax(initializations.get('glorot_uniform')((1, self.config['{}_memory_dim'.format(self.prefix)])))
                init_c = initializations.get('glorot_uniform')((1, self.config['{}_hidden_dim'.format(self.prefix)]))
                return core, [init_w, init_r, init_c]

        if model   == 'RNN':
            self.core, self.init = _build_RNN()
        elif model == 'NTM':
            self.core, self.init = _build_NTM()
        else:
            raise NotImplementedError

        self._add(self.core)
        if model == 'RNN':
            for init in self.init:
                self._add(init)

        self.set_name(name)

    # *****************************************************************
    # For Theano inputs.

    def get_context(self, context):
        # get context if "use_context" is True
        info  = dict()
        # if self.use_contxt:
        if self.model == 'RNN':
            # context is a matrix (nb_samples, context_dim)
            info['C'] = context
            info['init_h'] = self.init[0](context)

        elif self.model == 'NTM':
            # context is a tensor (nb_samples, memory_dim, memory_width)
            info['M']       = context
            if self.config['bias_code']:
                raise NotImplementedError
            else:
                info['init_ww'] = T.repeat(self.init[0], context.shape[0], axis=0)
                info['init_wr'] = T.repeat(self.init[1], context.shape[0], axis=0)
                info['init_c']  = T.repeat(self.init[2], context.shape[0], axis=0)
        else:
            raise NotImplementedError
        return info

    def loop(self, X, X_mask, info=None, return_sequence=False, return_full=False):
        if self.model == 'NTM':
            info['return_full'] = return_full

        Z = self.core(X, X_mask, return_sequence=return_sequence, **info)
        self._monitoring()
        return Z

    def step(self, X, prev_info):
        # run one step of the Recurrence
        if self.model == 'RNN':
            out = self.core(X, one_step=True, **prev_info)
            next_state = out
            next_info  = {'init_h': out, 'C': prev_info['C']}
        elif self.model == 'NTM':
            out = self.core(X, one_step=True, **prev_info)
            next_state = out[3]
            next_info  = dict()
            next_info['M']       = out[0]
            next_info['init_ww'] = out[1]
            next_info['init_wr'] = out[2]
            next_info['init_c']  = out[3]
        else:
            raise NotImplementedError
        return next_state, next_info

    def build_(self):
        # build a sampler in theano function for sampling.
        if self.model == 'RNN':
            context   = T.matrix()  # theano variable.
            logger.info('compile the function: get_init_state')
            info      = self.get_context(context)
            self.get_init_state \
                      = theano.function([context], info['init_h'],
                                        name='get_init_state')

            # **************************************************** #
            context   = T.matrix()  # theano variable.
            prev_X    = T.matrix('prev_X', dtype='float32')
            prev_stat = T.matrix('prev_state', dtype='float32')
            prev_info = dict()
            prev_info['C']      = context
            prev_info['init_h'] = prev_stat

            next_stat, next_info \
                = self.step(prev_X, prev_info)

            logger.info('compile the function: sample_next_state')
            inputs  = [prev_X, prev_stat, context]
            outputs = next_stat
            self.sample_next_state = theano.function(inputs, outputs, name='sample_next_state')

        elif self.model == 'NTM':
            memory  = T.tensor3()  # theano variable
            logger.info('compile the funtion: get_init_state')
            info    = self.get_context(memory)

            self.get_init_wr = theano.function([memory], info['init_wr'], name='get_init_wr')
            self.get_init_ww = theano.function([memory], info['init_ww'], name='get_init_ww')
            self.get_init_c  = theano.function([memory], info['init_c'],  name='get_init_c')

            # **************************************************** #
            memory    = T.tensor3()  # theano variable
            prev_X    = T.matrix('prev_X',  dtype='float32')
            prev_ww   = T.matrix('prev_ww', dtype='float32')
            prev_wr   = T.matrix('prev_wr', dtype='float32')
            prev_stat = T.matrix('prev_stat', dtype='float32')
            prev_info = {'M': memory, 'init_ww': prev_ww, 'init_wr': prev_wr, 'init_c': prev_stat}
            logger.info('compile the function: sample_next_0123')

            next_stat, next_info = self.step(prev_X, prev_info)
            inputs    = [prev_X, prev_ww, prev_wr, memory, prev_stat]
            outputs   = [next_info['M'], next_info['init_ww'], next_info['init_wr'], next_stat]
            self.sample_next_state = theano.function(inputs, outputs, name='sample_next_state')

        else:
            raise NotImplementedError

        logger.info('done.')

    # *****************************************************************
    # For Numpy inputs.
    def get_init(self, context):
        info = dict()
        if self.model == 'RNN':
            info['init_h'] = self.get_init_state(context)
            info['C']      = context
        elif self.model == 'NTM':
            if hasattr(self, 'get_init_ww'):
                info['init_ww'] = self.get_init_ww(context)
            if hasattr(self, 'get_init_wr'):
                info['init_wr'] = self.get_init_wr(context)
            if hasattr(self, 'get_init_c'):
                info['init_c']  = self.get_init_c(context)

            info['M'] = context
        else:
            raise NotImplementedError

        return info

    def get_next_state(self, prev_X, prev_info):
        if self.model == 'RNN':
            next_state = self.sample_next_state(
                prev_X, prev_info['init_h'], prev_info['C'])

            next_info = dict()
            next_info['C'] = prev_info['C']
            next_info['init_h'] = next_state
        elif self.model == 'NTM':
            next_info  = dict()
            assert 'init_ww' in prev_info
            assert 'init_wr' in prev_info
            assert 'init_c'  in prev_info
            assert 'M'       in prev_info

            next_info['M'], next_info['init_ww'], \
            next_info['init_wr'], next_info['init_c'] = self.sample_next_state(
                prev_X, prev_info['init_ww'], prev_info['init_wr'],
                prev_info['M'], prev_info['init_c'])

            next_state = next_info['init_c']
        else:
            raise NotImplementedError

        return next_state, next_info


class Encoder(Model):
    """
    Recurrent Neural Network/Neural Turing Machine-based Encoder
    It is used to compute the context vector.
    """

    def __init__(self,
                 config, rng, prefix='enc',
                 mode='RNN', embed=None):
        """
        mode = RNN: use a RNN Encoder
        mode = NTM: use a NTM Encoder
        """
        super(Encoder, self).__init__()
        self.config = config
        self.rng    = rng
        self.prefix = prefix
        self.mode   = mode
        self.name   = prefix

        """
        Create all elements of the Encoder's Computational graph
        """
        # create Embedding layers
        logger.info("{}_create embedding layers.".format(self.prefix))
        if embed:
            self.Embed = embed
        else:
            self.Embed = Embedding(
                self.config['enc_voc_size'],
                self.config['enc_embedd_dim'],
                name="{}_embed".format(self.prefix))
            self._add(self.Embed)

        # create Recurrent Base
        logger.info("{}_create Recurrent layers.".format(self.prefix))
        if self.mode == 'RNN' and self.config['bidirectional']:
            self.Forward = RecurrentBase(self.config, model=self.mode, name='forward',
                                         prefix='enc', use_contxt=self.config['enc_use_contxt'])
            self.Bakward = RecurrentBase(self.config, model=self.mode, name='backward',
                                         prefix='enc', use_contxt=self.config['enc_use_contxt'])

            self._add(self.Forward)
            self._add(self.Bakward)
        else:
            self.Recurrence = RecurrentBase(self.config, model=self.mode, name='encoder',
                                            prefix='enc', use_contxt=self.config['enc_use_contxt'])
            self._add(self.Recurrence)

        # there is no readout layers for encoder.

    def build_encoder(self, source, context=None):
        """
        Build the Encoder Computational Graph
        """
        if self.mode == 'RNN':
            # we use a Recurrent Neural Network Encoder (GRU)
            if not self.config['bidirectional']:
                X, X_mask = self.Embed(source, True)
                info      = self.Recurrence.get_context(context)
                X_out = self.Recurrence.loop(X, X_mask, info, return_sequence=False)
            else:
                source_back = source[:, ::-1]
                X1, X1_mask = self.Embed(source, True)
                X2, X2_mask = self.Embed(source_back, True)

                info        = self.Forward.get_context(context)
                X_out1      = self.Forward.loop(X1, X1_mask, info, return_sequence=False)
                info        = self.Bakward.get_context(context)
                X_out2      = self.Bakward.loop(X2, X2_mask, info, return_sequence=False)
                # X_out       = T.concatenate([X_out1, X_out2], axis=1)
                X_out       = 0.5 * X_out1 + 0.5 * X_out2
        elif self.mode == 'NTM':
            if not self.config['bidirectional']:
                X, X_mask = self.Embed(source, True)
            else:
                source_back = source[:, ::-1]
                X1, X1_mask = self.Embed(source, True)
                X2, X2_mask = self.Embed(source_back, True)
                X           = T.concatenate([X1, X2], axis=1)
                X_mask      = T.concatenate([X1_mask, X2_mask], axis=1)

            info  = self.Recurrence.get_context(context)
            # X_out here is the extracted memorybook. which can be used as a the initial memory of NTM Decoder.
            X_out = self.Recurrence.loop(X, X_mask, info, return_sequence=False, return_full=True)[0]
        else:
            raise NotImplementedError

        self._monitoring()
        return X_out


class Decoder(Model):
    """
    Recurrent Neural Network-based Decoder.
    It is used for:
        (1) Evaluation: compute the probability P(Y|X)
        (2) Prediction: sample the best result based on P(Y|X)
        (3) Beam-search
        (4) Scheduled Sampling (how to implement it?)
    """

    def __init__(self,
                 config, rng, prefix='dec',
                 mode='RNN', embed=None):
        """
        mode = RNN: use a RNN Decoder
        mode = NTM: use a NTM Decoder (Neural Turing Machine)
        """
        super(Decoder, self).__init__()
        self.config = config
        self.rng    = rng
        self.prefix = prefix
        self.name   = prefix
        self.mode   = mode

        """
        Create all elements of the Decoder's computational graph.
        """
        # create Embedding layers
        logger.info("{}_create embedding layers.".format(self.prefix))
        if embed:
            self.Embed = embed
        else:
            self.Embed = Embedding(
                self.config['dec_voc_size'],
                self.config['dec_embedd_dim'],
                name="{}_embed".format(self.prefix))
            self._add(self.Embed)

        # create Recurrent Base.
        logger.info("{}_create Recurrent layers.".format(self.prefix))
        self.Recurrence = RecurrentBase(self.config, model=self.mode, name='decoder',
                                        prefix='dec', use_contxt=self.config['dec_use_contxt'])

        # create readout layers
        logger.info("_create Readout layers")

        # 1. hidden layers readout.
        self.hidden_readout = Dense(
            self.config['dec_hidden_dim'],
            self.config['output_dim']
            if self.config['deep_out']
            else self.config['dec_voc_size'],
            activation='linear',
            name="{}_hidden_readout".format(self.prefix)
        )

        # 2. previous word readout
        self.prev_word_readout = None
        if self.config['bigram_predict']:
            self.prev_word_readout = Dense(
                self.config['dec_embedd_dim'],
                self.config['output_dim']
                if self.config['deep_out']
                else self.config['dec_voc_size'],
                activation='linear',
                name="{}_prev_word_readout".format(self.prefix),
                learn_bias=False
            )

        # 3. context readout
        self.context_readout = None
        if self.config['context_predict']:
            self.context_readout = Dense(
                self.config['dec_contxt_dim'],
                self.config['output_dim']
                if self.config['deep_out']
                else self.config['dec_voc_size'],
                activation='linear',
                name="{}_context_readout".format(self.prefix),
                learn_bias=False
            )

        # option: deep output (maxout)
        if self.config['deep_out']:
            self.activ = Activation(config['deep_out_activ'])
            # self.dropout = Dropout(rng=self.rng, p=config['dropout'])
            self.output_nonlinear = [self.activ]  # , self.dropout]
            self.output = Dense(
                self.config['output_dim'] / 2
                if config['deep_out_activ'] == 'maxout2'
                else self.config['output_dim'],

                self.config['dec_voc_size'],
                activation='softmax',
                name="{}_output".format(self.prefix),
                learn_bias=False
            )
        else:
            self.output_nonlinear = []
            self.output = Activation('softmax')

        # registration:
        self._add(self.Recurrence)
        self._add(self.hidden_readout)
        self._add(self.context_readout)
        self._add(self.prev_word_readout)
        self._add(self.output)

        if self.config['deep_out']:
            self._add(self.activ)
        # self._add(self.dropout)

        logger.info("create decoder ok.")

    @staticmethod
    def _grab_prob(probs, X):
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    """
    Build the decoder for evaluation
    """
    def prepare_xy(self, target):
        # Word embedding
        Y, Y_mask = self.Embed(target, True)  # (nb_samples, max_len, embedding_dim)

        if self.config['use_input']:
            X = T.concatenate([alloc_zeros_matrix(Y.shape[0], 1, Y.shape[2]), Y[:, :-1, :]], axis=1)
        else:
            X = 0 * Y

        # option ## drop words.

        X_mask    = T.concatenate([T.ones((Y.shape[0], 1)), Y_mask[:, :-1]], axis=1)
        Count     = T.cast(T.sum(X_mask, axis=1), dtype=theano.config.floatX)
        return X, X_mask, Y, Y_mask, Count

    def build_decoder(self, target, context=None, return_count=False):
        """
        Build the Decoder Computational Graph
        """
        X, X_mask, Y, Y_mask, Count = self.prepare_xy(target)
        info  = self.Recurrence.get_context(context)
        X_out = self.Recurrence.loop(X, X_mask, info=info, return_sequence=True)

        # Readout
        readout = self.hidden_readout(X_out)

        if self.config['context_predict']:
            # warning: only supports RNN, cannot supports Memory
            readout += self.context_readout(context).dimshuffle(0, 'x', 1) \

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        prob_dist = self.output(readout)  # (nb_samples, max_len, vocab_size)

        # log_old  = T.sum(T.log(self._grab_prob(prob_dist, target)), axis=1)
        log_prob = T.sum(T.log(self._grab_prob(prob_dist, target)) * X_mask, axis=1)
        log_ppl  = log_prob / Count

        self._monitoring()

        if return_count:
            return log_prob, Count
        else:
            return log_prob, log_ppl

    """
    Sampling Functions.
    """
    def _step_embed(self, prev_word):
        # word embedding (note that for the first word, embedding should be all zero)
        if self.config['use_input']:
            X = T.switch(
                prev_word[:, None] < 0,
                alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim']),
                self.Embed(prev_word)
            )
        else:
            X = alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim'])

        return X

    def _step_sample(self, X, next_stat, context):
        # compute the readout probability distribution and sample it
        # here the readout is a matrix, different from the learner.
        readout = self.hidden_readout(next_stat)

        if context.ndim == 2 and self.config['context_predict']:
            # warning: only supports RNN, cannot supports Memory
            readout += self.context_readout(context)

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        next_prob = self.output(readout)
        next_sample = self.rng.multinomial(pvals=next_prob).argmax(1)
        return next_prob, next_sample

    """
    Build the sampler for sampling/greedy search/beam search
    """

    def build_sampler(self):
        """
        Build a sampler which only steps once.
        Typically it only works for one word a time?
        """
        prev_word = T.vector('prev_word', dtype='int64')
        prev_X    = self._step_embed(prev_word)
        self.prev_embed = theano.function([prev_word], prev_X)

        self.Recurrence.build_()

        prev_X    = T.matrix('prev_X', dtype='float32')
        next_stat = T.matrix('next_state', dtype='float32')
        logger.info('compile the function: sample_next')

        if self.config['mode'] == 'RNN':
            context   = T.matrix('context')
        else:
            context   = T.tensor3('memory')

        next_prob, next_sample = self._step_sample(prev_X, next_stat, context)
        self.sample_next = theano.function([prev_X, next_stat, context],
                                           [next_prob, next_sample],
                                           name='sample_next',
                                           on_unused_input='warn')

        logger.info('done')

    """
    Generate samples, either with stochastic sampling or beam-search!
    """

    def get_sample(self, context, k=1, maxlen=30, stochastic=True, argmax=False):
        # beam size
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling!!'

        # prepare for searching
        sample = []
        score  = []
        if stochastic:
            score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype(theano.config.floatX)
        hyp_states = []
        hyp_infos  = []

        # get initial state of decoder Recurrence
        next_info  = self.Recurrence.get_init(context)
        # print 'sample with memory:\t', next_info['M'][0]
        # next_state = next_info['init_h']
        next_word  = -1 * np.ones((1,)).astype('int64')  # indicator for the first target word (bos target)
        print '<0e~k>'
        # Start searching!
        for ii in xrange(maxlen):
            # print next_word
            ctx = np.tile(context, [live_k, 1])
            next_embedding        = self.prev_embed(next_word)
            next_state, next_info = self.Recurrence.get_next_state(next_embedding, next_info)
            next_prob, next_word  = self.sample_next(next_embedding, next_state, ctx)  # wtf.

            if stochastic:
                # using stochastic sampling (or greedy sampling.)
                if argmax:
                    nw = next_prob[0].argmax()
                    next_word[0] = nw
                else:
                    nw = next_word[0]

                sample.append(nw)
                score += next_prob[0, nw]

                if nw == 0:  # sample reached the end
                    break

            else:
                # using beam-search
                # we can only computed in a flatten way!
                # Recently beam-search does not support NTM !!

                cand_scores = hyp_scores[:, None] - np.log(next_prob)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k - dead_k)]

                # fetch the best results.
                voc_size = next_prob.shape[1]
                trans_index = ranks_flat / voc_size
                word_index = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                # get the new hyp samples
                new_hyp_samples = []
                new_hyp_scores = np.zeros(k - dead_k).astype(theano.config.floatX)
                new_hyp_states = []
                new_hyp_infos  = {w: [] for w in next_info}

                for idx, [ti, wi] in enumerate(zip(trans_index, word_index)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[ti]))

                    for w in next_info:
                        new_hyp_infos[w].append(copy.copy(next_info[w][ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                hyp_infos  = {w: [] for w in next_info}

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_states[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                        for w in next_info:
                            hyp_infos[w].append(copy.copy(new_hyp_infos[w][ti]))

                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_word = np.array([w[-1] for w in hyp_samples])
                next_state = np.array(hyp_states)
                for w in hyp_infos:
                    next_info[w] = np.array(hyp_infos[w])
                pass
            pass

        # end.
        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    score.append(hyp_scores[idx])

        return sample, score


class RNNLM(Model):
    """
    RNN-LM, with context vector = 0.
    It is very similar with the implementation of VAE.
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name   = 'rnnlm'

    def build_(self):
        logger.info("build the RNN/NTM-decoder")
        self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)

        # registration:
        self._add(self.decoder)

        # objectives and optimizers
        self.optimizer = optimizers.get('adadelta')

        # saved the initial memories
        self.memory    = initializations.get('glorot_uniform')(
                    (self.config['dec_memory_dim'], self.config['dec_memory_wdth']))

        logger.info("create the RECURRENT language model. ok")

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

        # questions (theano variables)
        inputs  = T.imatrix()  # padded input word sequence (for training)
        if self.config['mode']   == 'RNN':
            context = alloc_zeros_matrix(inputs.shape[0], self.config['dec_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # decoding.
        target  = inputs
        logPxz, logPPL = self.decoder.build_decoder(target, context)

        # reconstruction loss
        loss_rec = T.mean(-logPxz)
        loss_ppl = T.exp(T.mean(-logPPL))

        L1       = T.sum([T.sum(abs(w)) for w in self.params])
        loss     = loss_rec

        updates = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs]

        self.train_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_fun')
        logger.info("pre-training functions compile done.")

        # add monitoring:
        self.monitor['context'] = context
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)

    def compile_train_CE(self):
        pass

    def compile_sample(self):
        # context vectors (as)
        self.decoder.build_sampler()
        logger.info("display functions compile done.")

    def compile_inference(self):
        pass

    def default_context(self):
        if self.config['mode'] == 'RNN':
            return np.zeros(shape=(1, self.config['dec_contxt_dim']), dtype=theano.config.floatX)
        elif self.config['mode'] == 'NTM':
            memory = self.memory.get_value()
            memory = memory.reshape((1, memory.shape[0], memory.shape[1]))
            return memory

    def generate_(self, context=None, mode='display', max_len=None):
        """
        :param action: action vector to guide the question.
                       If None, use a Gaussian to simulate the action.
        :return: question sentence in natural language.
        """
        # assert self.config['sample_stoch'], 'RNNLM sampling must be stochastic'
        # assert not self.config['sample_argmax'], 'RNNLM sampling cannot use argmax'

        if context is None:
            context = self.default_context()

        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'] if not max_len else max_len,
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None)

        sample, score = self.decoder.get_sample(context, **args)
        if not args['stochastic']:
            score = score / np.array([len(s) for s in sample])
            sample = sample[score.argmin()]
            score = score.min()
        else:
            score /= float(len(sample))

        return sample, np.exp(score)


class Helmholtz(RNNLM):
    """
    Helmholtz Machine as an probabilistic version AutoEncoder
    It is very similar with Variational Auto-Encoder
    We implement the Helmholtz RNN as well as Helmholtz Turing Machine here.
    Reference:
        Reweighted Wake-Sleep
            http://arxiv.org/abs/1406.2751
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode='RNN'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name   = 'helmholtz'

    def build_(self):
        logger.info("build the Helmholtz auto-encoder")
        if self.mode == 'NTM':
            assert self.config['enc_memory_dim']  == self.config['dec_memory_dim']
            assert self.config['enc_memory_wdth'] == self.config['dec_memory_wdth']

        self.encoder = Encoder(self.config, self.rng, prefix='enc', mode=self.mode)
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   embed=self.encoder.Embed, mode=self.mode)
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)

        # registration
        self._add(self.encoder)
        self._add(self.decoder)

        # The main difference between VAE and HM is that we can use
        # a more flexible prior instead of Gaussian here.
        # for example, we use a sigmoid prior here.

        # prior distribution is a bias layer
        if self.mode == 'RNN':
            # here we first forcus on Helmholtz Turing Machine
            # Thus the RNN version will be copied from Dial-DRL projects.
            raise NotImplementedError

        elif self.mode == 'NTM':
            self.Prior  = MemoryLinear(
                self.config['enc_memory_dim'],
                self.config['enc_memory_wdth'],
                activation='sigmoid',
                name='prior_proj',
                has_input=False
            )

            self.Post   = MemoryLinear(
                self.config['enc_memory_dim'],
                self.config['enc_memory_wdth'],
                activation='sigmoid',
                name='post_proj',
                has_input=True
            )

            self.Trans  = MemoryLinear(
                self.config['enc_memory_dim'],
                self.config['enc_memory_wdth'],
                activation='linear',
                name='trans_proj',
                has_input=True
            )

            # registration
            self._add(self.Prior)
            self._add(self.Post)
            self._add(self.Trans)

        else:
            raise NotImplementedError

        # objectives and optimizers
        self.optimizer = optimizers.get(self.config['optimizer'])

        # saved the initial memories
        self.memory    = initializations.get('glorot_uniform')(
                    (self.config['dec_memory_dim'], self.config['dec_memory_wdth']))

        logger.info("create Helmholtz Machine. ok")

    def compile_train(self):
        # questions (theano variables)
        inputs         = T.imatrix()  # padded input word sequence (for training)
        batch_size     = inputs.shape[0]
        if self.config['mode']   == 'RNN':
            context    = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context    = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # encoding
        memorybook     = self.encoder.build_encoder(inputs, context)

        # get Q(a|y) = sigmoid
        q_dis          = self.Post(memorybook)

        # repeats
        L              = self.config['repeats']
        target         = T.repeat(inputs[:, None, :],
                                  L,
                                  axis=1).reshape((inputs.shape[0] * L, inputs.shape[1]))
        q_dis          = T.repeat(q_dis[:, None, :, :],
                                  L,
                                  axis=1).reshape((q_dis.shape[0] * L, q_dis.shape[1], q_dis.shape[2]))

        # sample actions
        u              = self.rng.uniform(q_dis.shape)
        action         = T.cast(u <= q_dis, dtype=theano.config.floatX)

        # compute the exact probability for actions
        logQax         = action * T.log(q_dis) + (1 - action) * T.log(1 - q_dis)
        logQax         = logQax.sum(axis=-1).sum(axis=-1)

        # decoding.
        memorybook2    = self.Trans(action)
        logPxa, count  = self.decoder.build_decoder(target, memorybook2, return_count=True)

        # prior.
        p_dis          = self.Prior()
        logPa          = action * T.log(p_dis) + (1 - action) * T.log(1 - p_dis)
        logPa          = logPa.sum(axis=-1).sum(axis=-1)

        """
        Compute the weights
        """
        # reshape
        logQax         = logQax.reshape((batch_size, L))
        logPa          = logPa.reshape((batch_size, L))
        logPxa         = logPxa.reshape((batch_size, L))

        logPx_a        = logPa + logPxa

        # normalizing the weights
        log_wk         = logPx_a - logQax
        log_bpk        = logPa - logQax

        log_w_sum      = logSumExp(log_wk, axis=1)
        log_bp_sum     = logSumExp(log_bpk, axis=1)

        log_wnk        = log_wk - log_w_sum
        log_bpnk       = log_bpk - log_bp_sum

        # unbiased log-likelihood estimator
        logPx          = T.mean(log_w_sum - T.log(L))
        perplexity     = T.exp(-T.mean((log_w_sum - T.log(L)) / count))

        """
        Compute the Loss function
        """
        # loss    = weights * log [p(a)p(x|a)/q(a|x)]
        weights        = T.exp(log_wnk)
        bp             = T.exp(log_bpnk)
        bq             = 1. / L
        ess            = T.mean(1 / T.sum(weights ** 2, axis=1))

        factor         = self.config['factor']
        if self.config['variant_control']:
            lossQ   = -T.mean(T.sum(logQax * (weights - bq), axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(logPa  * (weights - bp), axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))          # log p(x|a)
            lossP   = lossPxa + lossPa

            updates = self.optimizer.get_updates(self.params, [lossP + factor * lossQ, weights, bp])
        else:
            lossQ   = -T.mean(T.sum(logQax * weights, axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(logPa  * weights, axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))   # log p(x|a)
            lossP   = lossPxa + lossPa

            updates = self.optimizer.get_updates(self.params, [lossP + factor * lossQ, weights])

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs]

        self.train_    = theano.function(train_inputs,
                                         [lossPa, lossPxa, lossQ, perplexity, ess],
                                         updates=updates,
                                         name='train_fun')

        logger.info("pre-training functions compile done.")

    def compile_sample(self):
        # # for Typical Auto-encoder, only conditional generation is useful.
        # inputs        = T.imatrix()  # padded input word sequence (for training)
        # if self.config['mode']   == 'RNN':
        #     context   = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        # elif self.config['mode'] == 'NTM':
        #     context   = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        # else:
        #     raise NotImplementedError
        # pass

        # sample the memorybook
        p_dis         = self.Prior()
        l             = T.iscalar()
        u             = self.rng.uniform((l, p_dis.shape[-2], p_dis.shape[-1]))
        binarybook    = T.cast(u <= p_dis, dtype=theano.config.floatX)
        memorybook    = self.Trans(binarybook)

        self.take     = theano.function([l], [binarybook, memorybook], name='take_action')

        # compile the sampler.
        self.decoder.build_sampler()
        logger.info('sampler function compile done.')

    def compile_inference(self):
        """
        build the hidden action prediction.
        """
        inputs         = T.imatrix()  # padded input word sequence (for training)

        if self.config['mode']   == 'RNN':
            context    = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context    = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # encoding
        memorybook     = self.encoder.build_encoder(inputs, context)

        # get Q(a|y) = sigmoid(.|Posterior * encoded)
        q_dis          = self.Post(memorybook)
        p_dis          = self.Prior()

        self.inference_ = theano.function([inputs], [memorybook, q_dis, p_dis])
        logger.info("inference function compile done.")

    def default_context(self):
        return self.take(1)[-1]



class BinaryHelmholtz(RNNLM):
    """
    Helmholtz Machine as an probabilistic version AutoEncoder
    It is very similar with Variational Auto-Encoder
    We implement the Helmholtz RNN as well as Helmholtz Turing Machine here.
    Reference:
        Reweighted Wake-Sleep
            http://arxiv.org/abs/1406.2751
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode='RNN'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name   = 'helmholtz'

    def build_(self):
        logger.info("build the Binary-Helmholtz auto-encoder")
        if self.mode == 'NTM':
            assert self.config['enc_memory_dim']  == self.config['dec_memory_dim']
            assert self.config['enc_memory_wdth'] == self.config['dec_memory_wdth']

        self.encoder = Encoder(self.config, self.rng, prefix='enc', mode=self.mode)
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   embed=self.encoder.Embed, mode=self.mode)
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)

        # registration
        self._add(self.encoder)
        self._add(self.decoder)

        # The main difference between VAE and HM is that we can use
        # a more flexible prior instead of Gaussian here.
        # for example, we use a sigmoid prior here.

        # prior distribution is a bias layer
        if self.mode == 'RNN':
            # here we first forcus on Helmholtz Turing Machine
            # Thus the RNN version will be copied from Dial-DRL projects.
            raise NotImplementedError

        elif self.mode == 'NTM':
            self.Prior  = MemoryLinear(
                self.config['enc_memory_dim'],
                self.config['enc_memory_wdth'],
                activation='sigmoid',
                name='prior_proj',
                has_input=False
            )

            # registration
            self._add(self.Prior)
        else:
            raise NotImplementedError

        # objectives and optimizers
        self.optimizer = optimizers.get(self.config['optimizer'])

        # saved the initial memories
        self.memory    = T.nnet.sigmoid(initializations.get('glorot_uniform')(
                    (self.config['dec_memory_dim'], self.config['dec_memory_wdth'])))

        logger.info("create Helmholtz Machine. ok")

    def compile_train(self):
        # questions (theano variables)
        inputs         = T.imatrix()  # padded input word sequence (for training)
        batch_size     = inputs.shape[0]
        if self.config['mode']   == 'RNN':
            context    = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context    = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # encoding
        memorybook     = self.encoder.build_encoder(inputs, context)

        # get Q(a|y) = sigmoid
        q_dis          = memorybook

        # repeats
        L              = self.config['repeats']
        target         = T.repeat(inputs[:, None, :],
                                  L,
                                  axis=1).reshape((inputs.shape[0] * L, inputs.shape[1]))
        q_dis          = T.repeat(q_dis[:, None, :, :],
                                  L,
                                  axis=1).reshape((q_dis.shape[0] * L, q_dis.shape[1], q_dis.shape[2]))

        # sample actions
        u              = self.rng.uniform(q_dis.shape)
        action         = T.cast(u <= q_dis, dtype=theano.config.floatX)

        # compute the exact probability for actions
        logQax         = action * T.log(q_dis) + (1 - action) * T.log(1 - q_dis)
        logQax         = logQax.sum(axis=-1).sum(axis=-1)

        # decoding.
        memorybook2    = action
        logPxa, count  = self.decoder.build_decoder(target, memorybook2, return_count=True)

        # prior.
        p_dis          = self.Prior()
        logPa          = action * T.log(p_dis) + (1 - action) * T.log(1 - p_dis)
        logPa          = logPa.sum(axis=-1).sum(axis=-1)

        """
        Compute the weights
        """
        # reshape
        logQax         = logQax.reshape((batch_size, L))
        logPa          = logPa.reshape((batch_size, L))
        logPxa         = logPxa.reshape((batch_size, L))

        logPx_a        = logPa + logPxa

        # normalizing the weights
        log_wk         = logPx_a - logQax
        log_bpk        = logPa - logQax

        log_w_sum      = logSumExp(log_wk, axis=1)
        log_bp_sum     = logSumExp(log_bpk, axis=1)

        log_wnk        = log_wk - log_w_sum
        log_bpnk       = log_bpk - log_bp_sum

        # unbiased log-likelihood estimator
        logPx          = T.mean(log_w_sum - T.log(L))
        perplexity     = T.exp(-T.mean((log_w_sum - T.log(L)) / count))

        """
        Compute the Loss function
        """
        # loss    = weights * log [p(a)p(x|a)/q(a|x)]
        weights        = T.exp(log_wnk)
        bp             = T.exp(log_bpnk)
        bq             = 1. / L
        ess            = T.mean(1 / T.sum(weights ** 2, axis=1))

        factor         = self.config['factor']
        if self.config['variant_control']:
            lossQ   = -T.mean(T.sum(logQax * (weights - bq), axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(logPa  * (weights - bp), axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))          # log p(x|a)
            lossP   = lossPxa + lossPa

            updates = self.optimizer.get_updates(self.params, [lossP + factor * lossQ, weights, bp])
        else:
            lossQ   = -T.mean(T.sum(logQax * weights, axis=1))   # log q(a|x)
            lossPa  = -T.mean(T.sum(logPa  * weights, axis=1))   # log p(a)
            lossPxa = -T.mean(T.sum(logPxa * weights, axis=1))   # log p(x|a)
            lossP   = lossPxa + lossPa

            updates = self.optimizer.get_updates(self.params, [lossP + factor * lossQ, weights])

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs]

        self.train_    = theano.function(train_inputs,
                                         [lossPa, lossPxa, lossQ, perplexity, ess],
                                         updates=updates,
                                         name='train_fun')

        logger.info("pre-training functions compile done.")

    def compile_sample(self):
        # # for Typical Auto-encoder, only conditional generation is useful.
        # inputs        = T.imatrix()  # padded input word sequence (for training)
        # if self.config['mode']   == 'RNN':
        #     context   = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        # elif self.config['mode'] == 'NTM':
        #     context   = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        # else:
        #     raise NotImplementedError
        # pass

        # sample the memorybook
        p_dis         = self.Prior()
        l             = T.iscalar()
        u             = self.rng.uniform((l, p_dis.shape[-2], p_dis.shape[-1]))
        binarybook    = T.cast(u <= p_dis, dtype=theano.config.floatX)

        self.take     = theano.function([l], binarybook, name='take_action')

        # compile the sampler.
        self.decoder.build_sampler()
        logger.info('sampler function compile done.')

    def compile_inference(self):
        """
        build the hidden action prediction.
        """
        inputs         = T.imatrix()  # padded input word sequence (for training)

        if self.config['mode']   == 'RNN':
            context    = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context    = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # encoding
        memorybook     = self.encoder.build_encoder(inputs, context)

        # get Q(a|y) = sigmoid(.|Posterior * encoded)
        q_dis          = memorybook
        p_dis          = self.Prior()

        self.inference_ = theano.function([inputs], [memorybook, q_dis, p_dis])
        logger.info("inference function compile done.")

    def default_context(self):
        return self.take(1)



class AutoEncoder(RNNLM):
    """
    Regular Auto-Encoder: RNN Encoder/Decoder
    Regular Neural Turing Machine
    """

    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name   = 'autoencoder'

    def build_(self):
        logger.info("build the RNN/NTM auto-encoder")
        self.encoder = Encoder(self.config, self.rng, prefix='enc', mode=self.mode)
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec',
                                   embed=self.encoder.Embed, mode=self.mode)
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)


        # registration
        self._add(self.encoder)
        self._add(self.decoder)

        # objectives and optimizers
        self.optimizer = optimizers.get(self.config['optimizer'])

        # saved the initial memories
        self.memory    = initializations.get('glorot_uniform')(
                    (self.config['dec_memory_dim'], self.config['dec_memory_wdth']))

        logger.info("create Autoencoder Network. ok")

    def compile_train(self, mode='train'):
        # questions (theano variables)
        inputs      = T.imatrix()  # padded input word sequence (for training)
        if self.config['mode']   == 'RNN':
            context    = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context    = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # encoding
        memorybook     = self.encoder.build_encoder(inputs, context)

        # decoding.
        target         = inputs
        logPxz, logPPL = self.decoder.build_decoder(target, memorybook)

        # reconstruction loss
        loss_rec       = T.mean(-logPxz)
        loss_ppl       = T.exp(T.mean(-logPPL))

        loss           = loss_rec
        updates        = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")
        train_inputs   = [inputs]

        self.train_    = theano.function(train_inputs,
                                         [loss_rec, loss_ppl],
                                         updates=updates,
                                         name='train_fun')
        self.test      = theano.function(train_inputs,
                                         [loss_rec, loss_ppl],
                                         name='test_fun')
        logger.info("pre-training functions compile done.")

    def compile_sample(self):
        # for Typical Auto-encoder, only conditional generation is useful.
        inputs        = T.imatrix()  # padded input word sequence (for training)
        if self.config['mode']   == 'RNN':
            context   = alloc_zeros_matrix(inputs.shape[0], self.config['enc_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context   = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError
        pass

        # encoding
        memorybook    = self.encoder.build_encoder(inputs, context)
        self.memorize = theano.function([inputs], memorybook, name='memorize')

        # compile the sampler.
        self.decoder.build_sampler()
        logger.info('sampler function compile done.')
