__author__ = 'jiataogu'
import theano
import logging
import copy

from emolga.layers.recurrent import *
from emolga.layers.ntm_minibatch import Controller
from emolga.layers.embeddings import *
from emolga.layers.attention import *
from emolga.layers.highwayNet import *
from emolga.models.encdec import *
from core import Model

# theano.config.exception_verbosity = 'high'
logger = logging          #.getLogger(__name__)
RNN    = GRU              # change it here for other RNN models.


class PtrDecoder(Model):
    """
    RNN-Decoder for Pointer Networks
    """
    def __init__(self,
                 config, rng, prefix='ptrdec'):
        super(PtrDecoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix

        """
        Create all elements of the Decoder's computational graph.
        """
        # create Initialization Layers
        logger.info("{}_create initialization layers.".format(self.prefix))
        self.Initializer = Dense(
            config['ptr_contxt_dim'],
            config['ptr_hidden_dim'],
            activation='tanh',
            name="{}_init".format(self.prefix)
        )

        # create RNN cells
        logger.info("{}_create RNN cells.".format(self.prefix))
        self.RNN = RNN(
            self.config['ptr_embedd_dim'],
            self.config['ptr_hidden_dim'],
            self.config['ptr_contxt_dim'],
            name="{}_cell".format(self.prefix)
        )
        self._add(self.Initializer)
        self._add(self.RNN)

        # create readout layers
        logger.info("_create Attention-Readout layers")
        self.attender = Attention(
            self.config['ptr_hidden_dim'],
            self.config['ptr_source_dim'],
            self.config['ptr_middle_dim'],
            name='{}_attender'.format(self.prefix)
        )
        self._add(self.attender)

    @staticmethod
    def grab_prob(probs, X):
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    @staticmethod
    def grab_source(source, target):
        # source : (nb_samples, source_num, source_dim)
        # target : (nb_samples, target_num)
        assert source.ndim == 3

        batch_size = source.shape[0]
        source_num = source.shape[1]
        source_dim = source.shape[2]
        target_num = target.shape[1]

        source_flt = source.reshape((batch_size * source_num, source_dim))
        target_idx = (target + (T.arange(batch_size) * source_num)[:, None]).reshape((batch_size * target_num,))

        value      = source_flt[target_idx].reshape((batch_size, target_num, source_dim))
        return value

    def build_decoder(self,
                      inputs,
                      source, target,
                      smask=None, tmask=None, context=None):
        """
        Build the Pointer Network Decoder Computational Graph
        """
        # inputs : (nb_samples, source_num, ptr_embedd_dim)
        # source : (nb_samples, source_num, source_dim)
        # smask  : (nb_samples, source_num)
        # target : (nb_samples, target_num)
        # tmask  : (nb_samples, target_num)
        # context: (nb_sample, context_dim)

        # initialized hidden state.
        assert context is not None
        Init_h = self.Initializer(context)

        # target is the source inputs.
        X      = self.grab_source(inputs, target)  # (nb_samples, target_num, source_dim)
        X      = T.concatenate([alloc_zeros_matrix(X.shape[0], 1, X.shape[2]),
                                X[:, :-1, :]], axis=1)

        X      = X.dimshuffle((1, 0, 2))
        # tmask  = tmask.dimshuffle((1, 0))

        # eat by recurrent net
        def _recurrence(x, prev_h, c, s, s_mask):
            # RNN read-out
            x_out  = self.RNN(x, mask=None, C=c, init_h=prev_h, one_step=True)
            s_out  = self.attender(x_out, s, s_mask, return_log=True)
            return x_out, s_out

        outputs, _ = theano.scan(
            _recurrence,
            sequences=[X],
            outputs_info=[Init_h, None],
            non_sequences=[context, source, smask]
        )

        log_prob_dist = outputs[-1].dimshuffle((1, 0, 2))
        # tmask         = tmask.dimshuffle((1, 0))
        log_prob      = T.sum(self.grab_prob(log_prob_dist, target) * tmask, axis=1)
        return log_prob

    """
    Sample one step
    """
    def _step_sample(self, prev_idx, prev_stat,
                     context, inputs, source, smask):
        X = T.switch(
                prev_idx[:, None] < 0,
                alloc_zeros_matrix(prev_idx.shape[0], self.config['ptr_embedd_dim']),
                self.grab_source(inputs, prev_idx[:, None])
                )

        # one step RNN
        X_out = self.RNN(X, C=context, init_h=prev_stat, one_step=True)
        next_stat = X_out

        # compute the attention read-out
        next_prob = self.attender(X_out, source, smask)
        next_sample = self.rng.multinomial(pvals=next_prob).argmax(1)
        return next_prob, next_sample, next_stat

    def build_sampler(self):
        """
        Build a sampler which only steps once.
        """
        logger.info("build sampler ...")
        if self.config['sample_stoch'] and self.config['sample_argmax']:
            logger.info("use argmax search!")
        elif self.config['sample_stoch'] and (not self.config['sample_argmax']):
            logger.info("use stochastic sampling!")
        elif self.config['sample_beam'] > 1:
            logger.info("use beam search! (beam_size={})".format(self.config['sample_beam']))

        # initial state of our Decoder.
        context = T.matrix()       # theano variable.
        init_h  = self.Initializer(context)
        logger.info('compile the function: get_init_state')
        self.get_init_state \
            = theano.function([context], init_h, name='get_init_state')
        logger.info('done.')

        # sampler: 1 x 1
        prev_idx  = T.vector('prev_idx', dtype='int64')
        prev_stat = T.matrix('prev_state', dtype='float32')
        inputs    = T.tensor3()
        source    = T.tensor3()
        smask     = T.imatrix()

        next_prob, next_sample, next_stat \
            = self._step_sample(prev_idx, prev_stat, context,
                                inputs, source, smask)

        # next word probability
        logger.info('compile the function: sample_next')
        inputs = [prev_idx, prev_stat, context, inputs, source, smask]
        outputs = [next_prob, next_sample, next_stat]
        self.sample_next = theano.function(inputs, outputs, name='sample_next')
        logger.info('done')
        pass

    """
    Generate samples, either with stochastic sampling or beam-search!
    """

    def get_sample(self, context, inputs, source, smask,
                   k=1, maxlen=30, stochastic=True, argmax=False, fixlen=False):
        # beam size
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling!!'

        # fix length cannot use beam search
        # if fixlen:
        #     assert k == 1

        # prepare for searching
        sample = []
        score = []
        if stochastic:
            score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype(theano.config.floatX)
        hyp_states = []

        # get initial state of decoder RNN with context
        next_state = self.get_init_state(context)
        next_word = -1 * np.ones((1,)).astype('int64')  # indicator for the first target word (bos target)

        # Start searching!
        for ii in xrange(maxlen):
            # print next_word
            ctx = np.tile(context, [live_k, 1])
            ipt = np.tile(inputs,  [live_k, 1, 1])
            sor = np.tile(source,  [live_k, 1, 1])
            smk = np.tile(smask,   [live_k, 1])

            next_prob, next_word, next_state \
                = self.sample_next(next_word, next_state,
                                   ctx, ipt, sor, smk)  # wtf.

            if stochastic:
                # using stochastic sampling (or greedy sampling.)
                if argmax:
                    nw = next_prob[0].argmax()
                    next_word[0] = nw
                else:
                    nw = next_word[0]

                sample.append(nw)
                score += next_prob[0, nw]

                if (not fixlen) and (nw == 0):  # sample reached the end
                    break

            else:
                # using beam-search
                # we can only computed in a flatten way!
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

                for idx, [ti, wi] in enumerate(zip(trans_index, word_index)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []

                for idx in xrange(len(new_hyp_samples)):
                    if (new_hyp_states[idx][-1] == 0) and (not fixlen):
                        sample.append(new_hyp_samples[idx])
                        score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])

                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_word = np.array([w[-1] for w in hyp_samples])
                next_state = np.array(hyp_states)
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


class PointerDecoder(Model):
    """
    RNN-Decoder for Pointer Networks [version 2]
    Pointer to 2 place once a time.
    """
    def __init__(self,
                 config, rng, prefix='ptrdec'):
        super(PointerDecoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix

        """
        Create all elements of the Decoder's computational graph.
        """
        # create Initialization Layers
        logger.info("{}_create initialization layers.".format(self.prefix))
        self.Initializer = Dense(
            config['ptr_contxt_dim'],
            config['ptr_hidden_dim'],
            activation='tanh',
            name="{}_init".format(self.prefix)
        )

        # create RNN cells
        logger.info("{}_create RNN cells.".format(self.prefix))
        self.RNN = RNN(
            self.config['ptr_embedd_dim'],
            self.config['ptr_hidden_dim'],
            self.config['ptr_contxt_dim'],
            name="{}_cell".format(self.prefix)
        )
        self._add(self.Initializer)
        self._add(self.RNN)

        # create 2 attention heads
        logger.info("_create Attention-Readout layers")
        self.att_head = Attention(
            self.config['ptr_hidden_dim'],
            self.config['ptr_source_dim'],
            self.config['ptr_middle_dim'],
            name='{}_head_attender'.format(self.prefix)
        )
        self.att_tail = Attention(
            self.config['ptr_hidden_dim'],
            self.config['ptr_source_dim'],
            self.config['ptr_middle_dim'],
            name='{}_tail_attender'.format(self.prefix)
        )

        self._add(self.att_head)
        self._add(self.att_tail)

    @staticmethod
    def grab_prob(probs, X):
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    @staticmethod
    def grab_source(source, target):
        # source : (nb_samples, source_num, source_dim)
        # target : (nb_samples, target_num)
        assert source.ndim == 3

        batch_size = source.shape[0]
        source_num = source.shape[1]
        source_dim = source.shape[2]
        target_num = target.shape[1]

        source_flt = source.reshape((batch_size * source_num, source_dim))
        target_idx = (target + (T.arange(batch_size) * source_num)[:, None]).reshape((batch_size * target_num,))

        value      = source_flt[target_idx].reshape((batch_size, target_num, source_dim))
        return value

    def build_decoder(self,
                      inputs,
                      source, target,
                      smask=None, tmask=None, context=None):
        """
        Build the Pointer Network Decoder Computational Graph
        """
        # inputs : (nb_samples, source_num, ptr_embedd_dim)
        # source : (nb_samples, source_num, source_dim)
        # smask  : (nb_samples, source_num)
        # target : (nb_samples, target_num)
        # tmask  : (nb_samples, target_num)
        # context: (nb_sample, context_dim)

        # initialized hidden state.
        assert context is not None
        Init_h = self.Initializer(context)

        # target is the source inputs.
        X      = self.grab_source(inputs, target)  # (nb_samples, target_num, source_dim)

        nb_dim = X.shape[0]
        tg_num = X.shape[1]
        sc_dim = X.shape[2]

        # since it changes to two pointers once a time:
        # concatenate + reshape
        def _get_ht(A, mask=False):
            if A.ndim == 2:
                B = A[:, -1:]
                if mask:
                    B *= 0.
                A = T.concatenate([A, B], axis=1)
                return A[:, ::2], A[:, 1::2]
            else:
                B = A[:, -1:, :]
                print B.ndim
                if mask:
                    B *= 0.
                A = T.concatenate([A, B], axis=1)
                return A[:, ::2, :], A[:, 1::2, :]

        Xh, Xt = _get_ht(X)
        Th, Tt = _get_ht(target)
        Mh, Mt = _get_ht(tmask, mask=True)

        Xa     = Xh + Xt
        Xa     = T.concatenate([alloc_zeros_matrix(nb_dim, 1, sc_dim),
                                Xa[:, :-1, :, :]], axis=1)
        Xa     = Xa.dimshuffle((1, 0, 2))

        # eat by recurrent net
        def _recurrence(x, prev_h, c, s, s_mask):
            # RNN read-out
            x_out  = self.RNN(x, mask=None, C=c, init_h=prev_h, one_step=True)
            h_out  = self.att_head(x_out, s, s_mask, return_log=True)
            t_out  = self.att_tail(x_out, s, s_mask, return_log=True)

            return x_out, h_out, t_out

        outputs, _ = theano.scan(
            _recurrence,
            sequences=[Xa],
            outputs_info=[Init_h, None, None],
            non_sequences=[context, source, smask]
        )
        log_prob_head = outputs[1].dimshuffle((1, 0, 2))
        log_prob_tail = outputs[2].dimshuffle((1, 0, 2))

        log_prob      = T.sum(self.grab_prob(log_prob_head, Th) * Mh, axis=1) \
                      + T.sum(self.grab_prob(log_prob_tail, Tt) * Mt, axis=1)
        return log_prob

    """
    Sample one step
    """
    def _step_sample(self,
                     prev_idx_h, prev_idx_t,
                     prev_stat,
                     context, inputs, source, smask):
        X = T.switch(
                prev_idx_h[:, None] < 0,
                alloc_zeros_matrix(prev_idx_h.shape[0], self.config['ptr_embedd_dim']),
                self.grab_source(inputs, prev_idx_h[:, None]) + self.grab_source(inputs, prev_idx_t[:, None])
                )

        # one step RNN
        X_out = self.RNN(X, C=context, init_h=prev_stat, one_step=True)
        next_stat = X_out

        # compute the attention read-out
        next_prob_h = self.att_head(X_out, source, smask)
        next_sample_h = self.rng.multinomial(pvals=next_prob_h).argmax(1)

        next_prob_t = self.att_tail(X_out, source, smask)
        next_sample_t = self.rng.multinomial(pvals=next_prob_t).argmax(1)
        return next_prob_h, next_sample_h, next_prob_t, next_sample_t, next_stat

    def build_sampler(self):
        """
        Build a sampler which only steps once.
        """
        logger.info("build sampler ...")
        if self.config['sample_stoch'] and self.config['sample_argmax']:
            logger.info("use argmax search!")
        elif self.config['sample_stoch'] and (not self.config['sample_argmax']):
            logger.info("use stochastic sampling!")
        elif self.config['sample_beam'] > 1:
            logger.info("use beam search! (beam_size={})".format(self.config['sample_beam']))

        # initial state of our Decoder.
        context = T.matrix()       # theano variable.
        init_h  = self.Initializer(context)
        logger.info('compile the function: get_init_state')
        self.get_init_state \
            = theano.function([context], init_h, name='get_init_state')
        logger.info('done.')

        # sampler: 1 x 1
        prev_idxh = T.vector('prev_idxh', dtype='int64')
        prev_idxt = T.vector('prev_idxt', dtype='int64')

        prev_stat = T.matrix('prev_state', dtype='float32')
        inputs    = T.tensor3()
        source    = T.tensor3()
        smask     = T.imatrix()

        next_prob_h, next_sample_h, next_prob_t, next_sample_t, next_stat \
            = self._step_sample(prev_idxh, prev_idxt, prev_stat, context,
                                inputs, source, smask)

        # next word probability
        logger.info('compile the function: sample_next')
        inputs = [prev_idxh, prev_idxt, prev_stat, context, inputs, source, smask]
        outputs = [next_prob_h, next_sample_h, next_prob_t, next_sample_t, next_stat]
        self.sample_next = theano.function(inputs, outputs, name='sample_next')
        logger.info('done')
        pass

    """
    Generate samples, either with stochastic sampling or beam-search!
    """

    def get_sample(self, context, inputs, source, smask,
                   k=1, maxlen=30, stochastic=True, argmax=False, fixlen=False):
        # beam size
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling!!'

        # fix length cannot use beam search
        # if fixlen:
        #     assert k == 1

        # prepare for searching
        sample = []
        score = []
        if stochastic:
            score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype(theano.config.floatX)
        hyp_states = []

        # get initial state of decoder RNN with context
        next_state = self.get_init_state(context)

        next_wordh = -1 * np.ones((1,)).astype('int64')  # indicator for the first target word (bos target)
        next_wordt = -1 * np.ones((1,)).astype('int64')

        # Start searching!
        for ii in xrange(maxlen):
            # print next_word
            ctx = np.tile(context, [live_k, 1])
            ipt = np.tile(inputs,  [live_k, 1, 1])
            sor = np.tile(source,  [live_k, 1, 1])
            smk = np.tile(smask,   [live_k, 1])

            next_probh, next_wordh, next_probt, next_wordt, next_state \
                = self.sample_next(next_wordh, next_wordt, next_state,
                                   ctx, ipt, sor, smk)  # wtf.

            if stochastic:
                # using stochastic sampling (or greedy sampling.)
                if argmax:
                    nw = next_probh[0].argmax()
                    next_wordh[0] = nw
                else:
                    nw = next_wordh[0]

                sample.append(nw)
                score += next_probh[0, nw]

                if (not fixlen) and (nw == 0):  # sample reached the end
                    break

                if argmax:
                    nw = next_probt[0].argmax()
                    next_wordt[0] = nw
                else:
                    nw = next_wordt[0]

                sample.append(nw)
                score += next_probt[0, nw]

                if (not fixlen) and (nw == 0):  # sample reached the end
                    break

            else:
                # using beam-search
                # I don't know how to apply 2 point beam-search
                # we can only computed in a flatten way!
                assert True, 'In this stage, I do not know how to use Beam-search for this problem.'

        return sample, score


class MemNet(Model):
    """
    Memory Networks:
        ==> Assign a Matrix to store rules
    """
    def __init__(self,
                 config, rng, learn_memory=False,
                 prefix='mem'):
        super(MemNet, self).__init__()
        self.config = config
        self.rng    = rng    # Theano random stream
        self.prefix = prefix
        self.init = initializations.get('glorot_uniform')

        if learn_memory:
            self.memory = self.init((self.config['mem_size'], self.config['mem_source_dim']))
            self.memory.name = '{}_inner_memory'.format(self.prefix)
            self.params += [self.memory]
        """
        Create the read-head of the MemoryNets
        """
        if self.config['mem_type'] == 'dnn':
            self.attender = Attention(
                config['mem_hidden_dim'],
                config['mem_source_dim'],
                config['mem_middle_dim'],
                name='{}_attender'.format(self.prefix)
            )
        else:
            self.attender = CosineAttention(
                config['mem_hidden_dim'],
                config['mem_source_dim'],
                use_pipe=config['mem_use_pipe'],
                name='{}_attender'.format(self.prefix)
            )
        self._add(self.attender)

    def __call__(self, key, memory=None, mem_mask=None, out_memory=None):
        # key:    (nb_samples, mem_hidden_dim)
        # memory: (nb_samples, mem_size, mem_source_dim)
        nb_samples = key.shape[0]
        if not memory:
            memory   = T.repeat(self.memory[None, :, :], nb_samples, axis=0)
            mem_mask = None

        if memory.ndim == 2:
            memory   = T.repeat(memory[None, :, :], nb_samples, axis=0)

        probout     = self.attender(key, memory, mem_mask)  # (nb_samples, mem_size)
        if self.config['mem_att_drop'] > 0:
            probout = T.clip(probout - self.config['mem_att_drop'], 0, 1)

        if out_memory is None:
            readout    = T.sum(memory * probout[:, :, None], axis=1)
        else:
            readout    = T.sum(out_memory * probout[:, :, None], axis=1)
        return readout, probout


class PtrNet(Model):
    """
    Pointer Networks [with/without] External Rule Memory
    """
    def __init__(self, config, n_rng, rng,
                 name='PtrNet', w_mem=True):
        super(PtrNet, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.name   = name
        self.w_mem  = w_mem

    def build_(self, encoder=None):
        logger.info("build the Pointer Networks")

        # encoder
        if not encoder:
            self.encoder = Encoder(self.config, self.rng, prefix='enc1')
            self._add(self.encoder)
        else:
            self.encoder = encoder

        if self.config['mem_output_mem']:
            self.encoder_out = Encoder(self.config, self.rng, prefix='enc_out')
            self._add(self.encoder_out)

        # twice encoding
        if self.config['ptr_twice_enc']:
            self.encoder2 = Encoder(self.config, self.rng, prefix='enc2', use_context=True)
            self._add(self.encoder2)

        # pointer decoder
        self.ptrdec  = PtrDecoder(self.config, self.rng)  # PtrDecoder(self.config, self.rng)
        self._add(self.ptrdec)

        # memory grabber
        self.grabber = MemNet(self.config, self.rng)
        self._add(self.grabber)

        # memory predictor :: alternative ::
        if self.config['use_predict']:
            logger.info('create a predictor AS Long Term Memory.s')
            if self.config['pred_type'] == 'highway':
                self.predictor = HighwayNet(self.config['mem_hidden_dim'],
                                            self.config['pred_depth'],
                                            activation='relu',
                                            name='phw')
            elif self.config['pred_type'] == 'dense':
                self.predictor = Dense(self.config['mem_hidden_dim'],
                                       self.config['mem_hidden_dim'],
                                       name='pdnn')
            elif self.config['pred_type'] == 'encoder':
                config = self.config
                # config['enc_embedd_dim'] = 300
                # config['enc_hidden_dim'] = 300
                self.predictor = Encoder(config, self.rng, prefix='enc3', use_context=False)
            else:
                NotImplementedError
            self._add(self.predictor)

        # objectives and optimizers
        assert self.config['optimizer'] == 'adam'
        self.optimizer = optimizers.get(self.config['optimizer'],
                                        kwargs=dict(rng=self.rng,
                                                    save=self.config['save_updates']))

    def build_train(self, memory=None, out_memory=None, compile_train=False, guide=None):
        # training function for Pointer Networks
        indices  = T.imatrix()  # padded word indices (for training)
        target   = T.imatrix()  # target indices (leading to relative locations)
        tmask    = T.imatrix()  # target masks
        pmask    = T.cast(1 - T.eq(target[:, 0], 0), dtype='float32')

        assert memory is not None, 'we must have an input memory'
        if self.config['mem_output_mem']:
            assert out_memory is not None,  'we must have an output memory'

        # L1 of memory
        loss_mem  = T.sum(abs(T.mean(memory, axis=0)))

        # encoding
        if not self.config['ptr_twice_enc']:
            source, inputs, smask, tail = self.encoder.build_encoder(indices, None, return_embed=True, return_sequence=True)

            # grab memory
            readout, probout = self.grabber(tail, memory)

            if not self.config['use_tail']:
                tailx = tail * 0.0
            else:
                tailx = tail

            if not self.config['use_memory']:
                readout *= 0.0

            # concatenate
            context  = T.concatenate([tailx, readout], axis=1)

            # if predict ?
            # predictor: minimize || readout - predict ||^2
            if self.config['use_predict']:
                if self.config['pred_type'] == 'encoder':
                    predict = self.predictor.build_encoder(indices, None, return_sequence=False)
                else:
                    predict = self.predictor(tail)

                # reconstruction loss [note that we only compute loss for correct memory read.]
                loss_r   = 0.5 * T.sum(pmask * T.sum(T.sqr(predict - readout), axis=-1).reshape(pmask.shape)) / T.sum(pmask)

                # use predicted readout to compute loss
                contextz = T.concatenate([tailx, predict], axis=1)
                sourcez, inputsz, smaskz = source, inputs, smask
        else:
            tail = self.encoder.build_encoder(indices, None, return_sequence=False)

            # grab memory
            readout, probout = self.grabber(tail, memory, out_memory)

            # get PrtNet input
            if not self.config['use_tail']:
                tailx = tail * 0.0
            else:
                tailx = tail

            if not self.config['use_memory']:
                readout *= 0.0

            # concatenate
            context0  = T.concatenate([tailx, readout], axis=1)

            # twice encoding ?
            source, inputs, smask, context = self.encoder2.build_encoder(
                indices, context=context0, return_embed=True, return_sequence=True)

            # if predict ?
            # predictor: minimize | readout - predict ||^2
            if self.config['use_predict']:
                if self.config['pred_type'] == 'encoder':
                    predict = self.predictor.build_encoder(indices, None, return_sequence=False)
                else:
                    predict = self.predictor(tail)

                # reconstruction loss [note that we only compute loss for correct memory read.]
                loss_r   = 0.5 * T.sum(pmask * T.sum(T.sqr(predict - readout), axis=-1).reshape(pmask.shape)) / T.sum(pmask)
                dist     = T.sum(T.sum(T.sqr(tail - readout), axis=-1).reshape(pmask.shape) * pmask) / T.sum(pmask)
                # use predicted readout to compute loss
                context1 = T.concatenate([tailx, predict], axis=1)

                # twice encoding..
                sourcez, inputsz, smaskz, contextz = self.encoder2.build_encoder(
                indices, context=context1, return_embed=True, return_sequence=True)

        # pointer decoder & loss
        logProb  = self.ptrdec.build_decoder(inputs, source, target,
                                             smask, tmask, context)
        loss     = T.mean(-logProb)

        # if predict?
        if self.config['use_predict']:
            logProbz = self.ptrdec.build_decoder(
                    inputsz, sourcez, target, smaskz, tmask, contextz)
            loss_z   = -T.sum(pmask * logProbz.reshape(pmask.shape)) / T.sum(pmask)

        # if guidance ?
        if guide:
            # attention loss
            # >>>>>>>   BE CAUTION !!!  <<<<<<
            # guide vector may contains '-1' which needs a mask for that.
            mask   = T.ones_like(guide) * (1 - T.eq(guide, -1))
            loss_g = T.mean(
                        -T.sum(
                            T.log(PtrDecoder.grab_prob(probout[:, None, :], guide)),
                        axis=1).reshape(mask.shape) * mask
                    )

            # attention accuracy
            attend = probout.argmax(axis=1, keepdims=True)
            maxp   = T.sum(probout.max(axis=1).reshape(mask.shape) * mask) / T.cast(T.sum(mask), 'float32')
            error  = T.sum((abs(attend - guide) * mask) > 0) / T.cast(T.sum(mask), 'float32')

            if self.config['mem_learn_guide']:
                loss  += loss_g

            # loss += 0.1 * loss_mem

        if compile_train:
            train_inputs = [indices, target, tmask, memory]
            if guide:
                train_inputs += [guide]
            logger.info("compiling the compuational graph ::training function::")
            updates  = self.optimizer.get_updates(self.params, loss)
            self.train_ = theano.function(train_inputs, loss, updates=updates, name='train_sub')
            logger.info("training functions compile done.")

        # output the building results for Training
        outputs  = [loss]
        if guide:
            outputs += [maxp, error]
        outputs += [indices, target, tmask]
        if self.config['use_predict']:
            outputs += [loss_r, loss_z, dist, readout]

        return outputs

    def build_sampler(self, memory=None, out_mem=None):
        # training function for Pointer Networks
        indices  = T.imatrix()  # padded word indices (for training)

        # encoding
        if not self.config['ptr_twice_enc']:
            # encoding
            source, inputs, smask, tail = self.encoder.build_encoder(indices, None, return_embed=True, return_sequence=True)

            # grab memory
            readout, probout = self.grabber(tail, memory, out_mem)

            if not self.config['use_tail']:
                tail *= 0.0

            if not self.config['use_memory']:
                readout *= 0.0

            # concatenate
            context  = T.concatenate([tail, readout], axis=1)
        else:
            tail = self.encoder.build_encoder(indices, None, return_sequence=False)

            # grab memory
            readout, probout = self.grabber(tail, memory, out_mem)
            if not self.config['use_tail']:
                tail *= 0.0

            if not self.config['use_memory']:
                readout *= 0.0

            # concatenate
            context0  = T.concatenate([tail, readout], axis=1)

            # twice encoding ?
            source, inputs, smask, context = self.encoder2.build_encoder(
                indices, context=context0, return_embed=True, return_sequence=True)

        # monitoring
        self.monitor['attention_prob'] = probout
        self._monitoring()

        return context, source, smask, inputs, indices

    def build_predict_sampler(self):
        # training function for Pointer Networks
        indices  = T.imatrix()  # padded word indices (for training)
        flag     = True

        # encoding
        if not self.config['ptr_twice_enc']:
            # encoding
            source, inputs, smask, tail = self.encoder.build_encoder(indices, None, return_embed=True, return_sequence=True)

            # predict memory
            if self.config['pred_type'] == 'encoder':
                readout = self.predictor.build_encoder(indices, None, return_sequence=False)
            else:
                readout = self.predictor(tail)

            if not self.config['use_tail']:
                tail *= 0.0

            if not self.config['use_memory']:
                readout *= 0.0

            # concatenate
            context  = T.concatenate([tail, readout], axis=1)
        else:
            tail = self.encoder.build_encoder(indices, None, return_sequence=False)

            # predict memory
            if self.config['pred_type'] == 'encoder':
                readout = self.predictor.build_encoder(indices, None, return_sequence=False)
            else:
                readout = self.predictor(tail)

            if not self.config['use_tail']:
                tail *= 0.0

            if not self.config['use_memory']:
                readout *= 0.0

            # concatenate
            context0  = T.concatenate([tail, readout], axis=1)

            # twice encoding ?
            source, inputs, smask, context = self.encoder2.build_encoder(
                indices, context=context0, return_embed=True, return_sequence=True)

        return context, source, smask, inputs, indices

    def generate_(self, inputs, context, source, smask):
        args = dict(k=4, maxlen=5, stochastic=False, argmax=False)
        sample, score = self.ptrdec.get_sample(context, inputs, source, smask,
                                               **args)
        if not args['stochastic']:
            score = score / np.array([len(s) for s in sample])
            sample = sample[score.argmin()]
            score = score.min()
        else:
            score /= float(len(sample))

        return sample, np.exp(score)