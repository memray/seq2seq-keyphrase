import math

__author__ = 'jiataogu, memray'
import theano

import logging
import copy
import emolga.basic.objectives as objectives
import emolga.basic.optimizers as optimizers

from theano.compile.nanguardmode import NanGuardMode
from emolga.layers.core import Dropout, Dense, Dense2, Identity
from emolga.layers.recurrent import *
from emolga.layers.ntm_minibatch import Controller
from emolga.layers.embeddings import *
from emolga.layers.attention import *
from emolga.models.core import Model

from nltk.stem.porter import *

logger = logging.getLogger(__name__)
RNN    = GRU             # change it here for other RNN models.


########################################################################################################################
# Encoder/Decoder Blocks ::::
#
# Encoder Back-up
# class Encoder(Model):
#     """
#     Recurrent Neural Network-based Encoder
#     It is used to compute the context vector.
#     """
#
#     def __init__(self,
#                  config, rng, prefix='enc',
#                  mode='Evaluation', embed=None, use_context=False):
#         super(Encoder, self).__init__()
#         self.config = config
#         self.rng = rng
#         self.prefix = prefix
#         self.mode = mode
#         self.name = prefix
#         self.use_context = use_context
#
#         """
#         Create all elements of the Encoder's Computational graph
#         """
#         # create Embedding layers
#         logger.info("{}_create embedding layers.".format(self.prefix))
#         if embed:
#             self.Embed = embed
#         else:
#             self.Embed = Embedding(
#                 self.config['enc_voc_size'],
#                 self.config['enc_embedd_dim'],
#                 name="{}_embed".format(self.prefix))
#             self._add(self.Embed)
#
#         if self.use_context:
#             self.Initializer = Dense(
#                 config['enc_contxt_dim'],
#                 config['enc_hidden_dim'],
#                 activation='tanh',
#                 name="{}_init".format(self.prefix)
#             )
#             self._add(self.Initializer)
#
#         """
#         Encoder Core
#         """
#         if self.config['encoder'] == 'RNN':
#             # create RNN cells
#             if not self.config['bidirectional']:
#                 logger.info("{}_create RNN cells.".format(self.prefix))
#                 self.RNN = RNN(
#                     self.config['enc_embedd_dim'],
#                     self.config['enc_hidden_dim'],
#                     None if not use_context
#                     else self.config['enc_contxt_dim'],
#                     name="{}_cell".format(self.prefix)
#                 )
#                 self._add(self.RNN)
#             else:
#                 logger.info("{}_create forward RNN cells.".format(self.prefix))
#                 self.forwardRNN = RNN(
#                     self.config['enc_embedd_dim'],
#                     self.config['enc_hidden_dim'],
#                     None if not use_context
#                     else self.config['enc_contxt_dim'],
#                     name="{}_fw_cell".format(self.prefix)
#                 )
#                 self._add(self.forwardRNN)
#
#                 logger.info("{}_create backward RNN cells.".format(self.prefix))
#                 self.backwardRNN = RNN(
#                     self.config['enc_embedd_dim'],
#                     self.config['enc_hidden_dim'],
#                     None if not use_context
#                     else self.config['enc_contxt_dim'],
#                     name="{}_bw_cell".format(self.prefix)
#                 )
#                 self._add(self.backwardRNN)
#
#             logger.info("create encoder ok.")
#
#         elif self.config['encoder'] == 'WS':
#             # create weighted sum layers.
#             if self.config['ws_weight']:
#                 self.WS  = Dense(self.config['enc_embedd_dim'],
#                                  self.config['enc_hidden_dim'], name='{}_ws'.format(self.prefix))
#                 self._add(self.WS)
#
#             logger.info("create encoder ok.")
#
#     def build_encoder(self, source, context=None, return_embed=False):
#         """
#         Build the Encoder Computational Graph
#         """
#         # Initial state
#         Init_h = None
#         if self.use_context:
#             Init_h = self.Initializer(context)
#
#         # word embedding
#         if self.config['encoder'] == 'RNN':
#             if not self.config['bidirectional']:
#                 X, X_mask = self.Embed(source, True)
#                 if not self.config['pooling']:
#                     X_out = self.RNN(X, X_mask, C=context, init_h=Init_h, return_sequence=False)
#                 else:
#                     X_out = self.RNN(X, X_mask, C=context, init_h=Init_h, return_sequence=True)
#             else:
#                 source2 = source[:, ::-1]
#                 X,  X_mask = self.Embed(source, True)
#                 X2, X2_mask = self.Embed(source2, True)
#
#                 if not self.config['pooling']:
#                     X_out1 = self.backwardRNN(X, X_mask, C=context, init_h=Init_h, return_sequence=False)
#                     X_out2 = self.forwardRNN( X2, X2_mask, C=context, init_h=Init_h, return_sequence=False)
#                     X_out  = T.concatenate([X_out1, X_out2], axis=1)
#                 else:
#                     X_out1 = self.backwardRNN(X, X_mask, C=context, init_h=Init_h, return_sequence=True)
#                     X_out2 = self.forwardRNN( X2, X2_mask, C=context, init_h=Init_h, return_sequence=True)
#                     X_out  = T.concatenate([X_out1, X_out2], axis=2)
#
#             if self.config['pooling'] == 'max':
#                 X_out = T.max(X_out, axis=1)
#             elif self.config['pooling'] == 'mean':
#                 X_out = T.mean(X_out, axis=1)
#
#         elif self.config['encoder'] == 'WS':
#             X, X_mask = self.Embed(source, True)
#             if self.config['ws_weight']:
#                 X_out = T.sum(self.WS(X) * X_mask[:, :, None], axis=1) / T.sum(X_mask, axis=1, keepdims=True)
#             else:
#                 assert self.config['enc_embedd_dim'] == self.config['enc_hidden_dim'], \
#                     'directly sum should match the dimension'
#                 X_out = T.sum(X * X_mask[:, :, None], axis=1) / T.sum(X_mask, axis=1, keepdims=True)
#         else:
#             raise NotImplementedError
#
#         if return_embed:
#             return X_out, X, X_mask
#         return X_out
#
#     def compile_encoder(self, with_context=False):
#         source  = T.imatrix()
#         if with_context:
#             context = T.matrix()
#             self.encode = theano.function([source, context],
#                                           self.build_encoder(source, context))
#         else:
#             self.encode = theano.function([source],
#                                       self.build_encoder(source, None))

class Encoder(Model):
    """
    Recurrent Neural Network-based Encoder
    It is used to compute the context vector.
    """

    def __init__(self,
                 config, rng, prefix='enc',
                 mode='Evaluation', embed=None, use_context=False):
        super(Encoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix
        self.mode = mode
        self.name = prefix
        self.use_context = use_context

        self.return_embed = False
        self.return_sequence = False

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

        if self.use_context:
            self.Initializer = Dense(
                config['enc_contxt_dim'],
                config['enc_hidden_dim'],
                activation='tanh',
                name="{}_init".format(self.prefix)
            )
            self._add(self.Initializer)

        """
        Encoder Core
        """
        # create RNN cells
        if not self.config['bidirectional']:
            logger.info("{}_create RNN cells.".format(self.prefix))
            self.RNN = RNN(
                self.config['enc_embedd_dim'],
                self.config['enc_hidden_dim'],
                None if not use_context
                else self.config['enc_contxt_dim'],
                name="{}_cell".format(self.prefix)
            )
            self._add(self.RNN)
        else:
            logger.info("{}_create forward RNN cells.".format(self.prefix))
            self.forwardRNN = RNN(
                self.config['enc_embedd_dim'],
                self.config['enc_hidden_dim'],
                None if not use_context
                else self.config['enc_contxt_dim'],
                name="{}_fw_cell".format(self.prefix)
            )
            self._add(self.forwardRNN)

            logger.info("{}_create backward RNN cells.".format(self.prefix))
            self.backwardRNN = RNN(
                self.config['enc_embedd_dim'],
                self.config['enc_hidden_dim'],
                None if not use_context
                else self.config['enc_contxt_dim'],
                name="{}_bw_cell".format(self.prefix)
            )
            self._add(self.backwardRNN)

        logger.info("create encoder ok.")

    def build_encoder(self, source, context=None, return_embed=False, return_sequence=False):
        """
        Build the Encoder Computational Graph

        For the default configurations (with attention)
            with_context=False, return_sequence=True, return_embed=True
        Input:
            source : source text, a list of indexes [nb_sample * max_len]
            context: None
        Return:
            For Attention model:
                return_sequence=True: to return the embedding at each time, not just the end state
                return_embed=True:
                    X_out:  a list of vectors [nb_sample, max_len, 2*enc_hidden_dim], encoding of each time state (concatenate both forward and backward RNN)
                    X:      embedding of text X [nb_sample, max_len, enc_embedd_dim]
                    X_mask: mask, an array showing which elements in X are not 0 [nb_sample, max_len]
                    X_tail: encoding of ending of X, seems not make sense for bidirectional model (head+tail) [nb_sample, 2*enc_hidden_dim]
                there's bug on X_tail, but luckily we don't use it often

        nb_sample:  number of samples, defined by batch size
        max_len:    max length of sentence (should be same after padding)
        """
        # Initial state
        Init_h = None
        if self.use_context:
            Init_h = self.Initializer(context)

        # word embedding
        if not self.config['bidirectional']:
            X, X_mask = self.Embed(source, True)
            X_out     = self.RNN(X, X_mask, C=context, init_h=Init_h, return_sequence=return_sequence)
            if return_sequence:
                X_tail    = X_out[:, -1]
            else:
                X_tail    = X_out
        else:
            # reverse the source for backwardRNN
            source2 = source[:, ::-1]
            # map text to embedding
            X,  X_mask = self.Embed(source, True)
            X2, X2_mask = self.Embed(source2, True)

            # get the encoding at each time t. [Bug?] run forwardRNN on the reverse text?
            X_out1 = self.backwardRNN(X, X_mask,  C=context, init_h=Init_h, return_sequence=return_sequence)
            X_out2 = self.forwardRNN(X2, X2_mask, C=context, init_h=Init_h, return_sequence=return_sequence)

            # concatenate vectors of both forward and backward
            if not return_sequence:
                # [Bug]I think the X_out of backwardRNN is time 0, but for forwardRNN is ending time
                X_out  = T.concatenate([X_out1, X_out2], axis=1)
                X_tail = X_out
            else:
                # reverse the encoding of forwardRNN(actually backwardRNN), so the X_out is backward
                X_out  = T.concatenate([X_out1, X_out2[:, ::-1, :]], axis=2)
                # [Bug] X_out1[-1] is time 0, but X_out2[-1] is ending time
                X_tail = T.concatenate([X_out1[:, -1], X_out2[:, -1]], axis=1)

        X_mask  = T.cast(X_mask, dtype='float32')
        if return_embed:
            return X_out, X, X_mask, X_tail
        return X_out

    def compile_encoder(self, with_context=False, return_embed=False, return_sequence=False):
        source  = T.imatrix()
        self.return_embed = return_embed
        self.return_sequence = return_sequence
        if with_context:
            context = T.matrix()

            self.encode = theano.function([source, context],
                                          self.build_encoder(source, context,
                                                             return_embed=return_embed,
                                                             return_sequence=return_sequence))
        else:
            self.encode = theano.function([source],
                                          self.build_encoder(source, None,
                                                             return_embed=return_embed,
                                                             return_sequence=return_sequence))


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
                 mode='RNN', embed=None,
                 highway=False):
        """
        mode = RNN: use a RNN Decoder
        """
        super(Decoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix
        self.name = prefix
        self.mode = mode

        self.highway = highway
        self.init = initializations.get('glorot_uniform')
        self.sigmoid = activations.get('sigmoid')

        # use standard drop-out for input & output.
        # I believe it should not use for context vector.
        self.dropout = config['dropout']
        if self.dropout > 0:
            logger.info('Use standard-dropout!!!!')
            self.D   = Dropout(rng=self.rng, p=self.dropout, name='{}_Dropout'.format(prefix))

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

        # create Initialization Layers
        logger.info("{}_create initialization layers.".format(self.prefix))
        if not config['bias_code']:
            self.Initializer = Zero()
        else:
            self.Initializer = Dense(
                config['dec_contxt_dim'],
                config['dec_hidden_dim'],
                activation='tanh',
                name="{}_init".format(self.prefix)
            )

        # create RNN cells
        logger.info("{}_create RNN cells.".format(self.prefix))
        self.RNN = RNN(
            self.config['dec_embedd_dim'],
            self.config['dec_hidden_dim'],
            self.config['dec_contxt_dim'],
            name="{}_cell".format(self.prefix)
        )

        self._add(self.Initializer)
        self._add(self.RNN)

        # HighWay Gating
        if highway:
            logger.info("HIGHWAY CONNECTION~~~!!!")
            assert self.config['context_predict']
            assert self.config['dec_contxt_dim'] == self.config['dec_hidden_dim']

            self.C_x = self.init((self.config['dec_contxt_dim'],
                                  self.config['dec_hidden_dim']))
            self.H_x = self.init((self.config['dec_hidden_dim'],
                                  self.config['dec_hidden_dim']))
            self.b_x = initializations.get('zero')(self.config['dec_hidden_dim'])

            self.C_x.name = '{}_Cx'.format(self.prefix)
            self.H_x.name = '{}_Hx'.format(self.prefix)
            self.b_x.name = '{}_bx'.format(self.prefix)
            self.params += [self.C_x, self.H_x, self.b_x]

        # create readout layers
        logger.info("_create Readout layers")

        # 1. hidden layers readout.
        self.hidden_readout = Dense(
            self.config['dec_hidden_dim'],
            self.config['output_dim']
            if self.config['deep_out'] # what's deep out?
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
            if not self.config['leaky_predict']:
                self.context_readout = Dense(
                    self.config['dec_contxt_dim'],
                    self.config['output_dim']
                    if self.config['deep_out']
                    else self.config['dec_voc_size'],
                    activation='linear',
                    name="{}_context_readout".format(self.prefix),
                    learn_bias=False
                )
            else:
                assert self.config['dec_contxt_dim'] == self.config['dec_hidden_dim']
                self.context_readout = self.hidden_readout

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
        self._add(self.hidden_readout)

        if not self.config['leaky_predict']:
            self._add(self.context_readout)

        self._add(self.prev_word_readout)
        self._add(self.output)

        if self.config['deep_out']:
            self._add(self.activ)
        # self._add(self.dropout)

        logger.info("create decoder ok.")

    @staticmethod
    def _grab_prob(probs, X):
        '''
        return the predicted probabilities of target term
        :param probs:           [nb_samples, max_len_target, vocab_size], predicted probabilities of all terms(size=vocab_size) on each target position (size=max_len_target)
        :param X:               [nb_sample, max_len_target], contains the index of target term
        :return: probs_target:  [nb_sample, max_len_target], predicted probabilities of each target term
        '''
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]
        # reshape to a 2D list, axis0 is batch-term, axis1 is vocabulary
        probs = probs.reshape((batch_size * max_len, vocab_size))
        '''
        return the predicting probability of target term
              T.arange(batch_size * max_len) indicates the index of each prediction
              X.flatten(1), convert X into a 1-D list, index of target terms
              reshape(X.shape), reshape to X's shape [nb_sample, max_len_target]
        '''
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    """
    Build the decoder for evaluation
    """
    def prepare_xy(self, target):
        # Word embedding of target, mask is a list of [0,1] shows which elements are not zero
        Y, Y_mask = self.Embed(target, True)  # (nb_samples, max_len, embedding_dim)

        if self.config['use_input']:
            X = T.concatenate([alloc_zeros_matrix(Y.shape[0], 1, Y.shape[2]), Y[:, :-1, :]], axis=1)
        else:
            X = 0 * Y

        # option ## drop words.

        X_mask    = T.concatenate([T.ones((Y.shape[0], 1)), Y_mask[:, :-1]], axis=1)
        Count     = T.cast(T.sum(X_mask, axis=1), dtype=theano.config.floatX)
        return X, X_mask, Y, Y_mask, Count

    def build_decoder(self, target, context=None,
                      return_count=False,
                      train=True):

        """
        Build the Decoder Computational Graph
        For training/testing
        """
        X, X_mask, Y, Y_mask, Count = self.prepare_xy(target)

        # input drop-out if any.
        if self.dropout > 0:
            X = self.D(X, train=train)

        # Initial state of RNN
        Init_h = self.Initializer(context)
        if not self.highway:
            X_out  = self.RNN(X, X_mask, C=context, init_h=Init_h, return_sequence=True)

            # Readout
            readout = self.hidden_readout(X_out)
            if self.dropout > 0:
                readout = self.D(readout, train=train)

            if self.config['context_predict']:
                readout += self.context_readout(context).dimshuffle(0, 'x', 1)
        else:
            X      = X.dimshuffle((1, 0, 2))
            X_mask = X_mask.dimshuffle((1, 0))

            def _recurrence(x, x_mask, prev_h, c):
                # compute the highway gate for context vector.
                xx    = dot(c, self.C_x, self.b_x) + dot(prev_h, self.H_x)  # highway gate.
                xx    = self.sigmoid(xx)

                cy    = xx * c   # the path without using RNN
                x_out = self.RNN(x, mask=x_mask, C=c, init_h=prev_h, one_step=True)
                hx    = (1 - xx) * x_out
                return x_out, hx, cy

            outputs, _ = theano.scan(
                _recurrence,
                sequences=[X, X_mask],
                outputs_info=[Init_h, None, None],
                non_sequences=[context]
            )

            # hidden readout + context readout
            readout   = self.hidden_readout( outputs[1].dimshuffle((1, 0, 2)))
            if self.dropout > 0:
                readout = self.D(readout, train=train)

            readout  += self.context_readout(outputs[2].dimshuffle((1, 0, 2)))

            # return to normal size.
            X      = X.dimshuffle((1, 0, 2))
            X_mask = X_mask.dimshuffle((1, 0))

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        prob_dist = self.output(readout)  # (nb_samples, max_len, vocab_size)
        # log_old  = T.sum(T.log(self._grab_prob(prob_dist, target)), axis=1)
        log_prob = T.sum(T.log(self._grab_prob(prob_dist, target)) * X_mask, axis=1)
        log_ppl  = log_prob / Count

        if return_count:
            return log_prob, Count
        else:
            return log_prob, log_ppl


    def _step_sample(self, prev_word, prev_stat, context):
        """
        Sample one step
        """

        # word embedding (note that for the first word, embedding should be all zero)
        if self.config['use_input']:
            X = T.switch(
                prev_word[:, None] < 0,
                alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim']),
                self.Embed(prev_word)
            )
        else:
            X = alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim'])

        if self.dropout > 0:
            X = self.D(X, train=False)

        # apply one step of RNN
        if not self.highway:
            X_proj = self.RNN(X, C=context, init_h=prev_stat, one_step=True)
            next_stat = X_proj

            # compute the readout probability distribution and sample it
            # here the readout is a matrix, different from the learner.
            readout = self.hidden_readout(next_stat)
            if self.dropout > 0:
                readout = self.D(readout, train=False)

            if self.config['context_predict']:
                readout += self.context_readout(context)
        else:
            xx     = dot(context, self.C_x, self.b_x) + dot(prev_stat, self.H_x)  # highway gate.
            xx     = self.sigmoid(xx)

            X_proj = self.RNN(X, C=context, init_h=prev_stat, one_step=True)
            next_stat = X_proj

            readout  = self.hidden_readout((1 - xx) * X_proj)
            if self.dropout > 0:
                readout = self.D(readout, train=False)

            readout += self.context_readout(xx * context)

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        next_prob = self.output(readout)
        next_sample = self.rng.multinomial(pvals=next_prob).argmax(1)
        return next_prob, next_sample, next_stat

    """
    Build the sampler for sampling/greedy search/beam search
    """

    def build_sampler(self):
        """
        Build a sampler which only steps once.
        Typically it only works for one word a time?
        """
        logger.info("build sampler ...")
        if self.config['sample_stoch'] and self.config['sample_argmax']:
            logger.info("use argmax search!")
        elif self.config['sample_stoch'] and (not self.config['sample_argmax']):
            logger.info("use stochastic sampling!")
        elif self.config['sample_beam'] > 1:
            logger.info("use beam search! (beam_size={})".format(self.config['sample_beam']))

        # initial state of our Decoder.
        context = T.matrix()  # theano variable.

        init_h = self.Initializer(context)
        logger.info('compile the function: get_init_state')
        self.get_init_state \
            = theano.function([context], init_h, name='get_init_state')
        logger.info('done.')

        # word sampler: 1 x 1
        prev_word = T.vector('prev_word', dtype='int64')
        prev_stat = T.matrix('prev_state', dtype='float32')

        next_prob, next_sample, next_stat \
            = self._step_sample(prev_word, prev_stat, context)

        # next word probability
        logger.info('compile the function: sample_next')
        inputs = [prev_word, prev_stat, context]
        outputs = [next_prob, next_sample, next_stat]

        self.sample_next = theano.function(inputs, outputs, name='sample_next')
        logger.info('done')
        pass

    """
    Build a Stochastic Sampler which can use SCAN to work on GPU.
    However it cannot be used in Beam-search.
    """

    def build_stochastic_sampler(self):
        context = T.matrix()
        init_h = self.Initializer(context)

        logger.info('compile the function: sample')
        pass

    """
    Generate samples, either with stochastic sampling or beam-search!
    """

    def get_sample(self, context, k=1, maxlen=30, stochastic=True, argmax=False, fixlen=False):
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
            # print(next_word)
            ctx = np.tile(context, [live_k, 1]) # copy context live_k times, into a list size=[live_k]
            next_prob, next_word, next_state \
                = self.sample_next(next_word, next_state, ctx)  # wtf.

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


class DecoderAtt(Decoder):
    """
    Recurrent Neural Network-based Decoder
    with Attention Mechanism
    """
    def __init__(self,
                 config, rng, prefix='dec',
                 mode='RNN', embed=None,
                 copynet=False, identity=False):
        super(DecoderAtt, self).__init__(
                config, rng, prefix,
                 mode, embed, False)

        self.copynet  = copynet
        self.identity = identity
        # attention reader
        self.attention_reader = Attention(
            self.config['dec_hidden_dim'],
            self.config['dec_contxt_dim'],
            1000,
            name='source_attention'
        )
        self._add(self.attention_reader)

        # if use copynet
        if self.copynet:
            if not self.identity:
                self.Is = Dense(
                    self.config['dec_contxt_dim'],
                    self.config['dec_embedd_dim'],
                    name='in-trans'
                )
            else:
                assert self.config['dec_contxt_dim'] == self.config['dec_embedd_dim']
                self.Is = Identity(name='ini')

            self.Os = Dense(
                self.config['dec_readout_dim'],
                self.config['dec_contxt_dim'],
                name='out-trans'
            )
            self._add(self.Is)
            self._add(self.Os)

        logger.info('adjust decoder ok.')

    def prepare_xy(self, target, context=None):
        """
        Build the decoder for evaluation

        We have padded the last column of target (target[-1]) to be 0s
        The Y and Y_mask here are embedding and mask of target.

        Here for training purpose, we create a new input X and X_mask,
                in which pad one column of 0s at the beginning as start signal, and delete the last column of 0s

        self.config['use_input']: True, if False, may mean that don't input the current word, only h_t = g(h_t-1)+h(0)
        :return X:              a matrix of 0s [nb_sample, 1, enc_embedd_dim], concatenate with Y [nb_sample, max_len-1, enc_embedd_dim]
                                result in [nb_sample, [0]+Y[:-1], enc_embedd_dim], first one is 0, and latter one is y[t-1]
        :return X_mask:         a matrix of 1s [nb_sample, 1],  concatenate with Y_mask [nb_sample, max_len-1]
                                result in [nb_sample, [0]+Y_mask[:-1]], first one is 0, and latter one is y_mask[t-1]

        :return Y:              embedding of target Y [nb_sample, max_len_target, enc_embedd_dim]
        :return Y_mask:         mask of target Y  [nb_sample, max_len_target]
        :return Count:          how many word in X, T.sum(X_mask, axis=1), is used for computing entropy and ppl
        """
        if not self.copynet:
            # Word embedding
            Y, Y_mask = self.Embed(target, True)  # (nb_samples, max_len, enc_embedd_dim)
        else:
            Y, Y_mask = self.Embed(target, True, context=self.Is(context))

        if self.config['use_input']:
            X = T.concatenate([alloc_zeros_matrix(Y.shape[0], 1, Y.shape[2]), Y[:, :-1, :]], axis=1)
        else:
            X = 0 * Y

        X_mask    = T.concatenate([T.ones((Y.shape[0], 1)), Y_mask[:, :-1]], axis=1)
        Count     = T.cast(T.sum(X_mask, axis=1), dtype=theano.config.floatX)
        return X, X_mask, Y, Y_mask, Count

    def build_decoder(self,
                      target,
                      context, c_mask,
                      return_count=False,
                      train=True):
        """
        Build the Computational Graph ::> Context is essential
        :param target:              index vector of target   [nb_sample, max_len]
        :param context:             encoding of source text, [nb_sample, max_len, 2*enc_hidden_dim]
        :param c_mask:              mask of source text
        :param return_count,train:  not used

        :return value of objective function
            log_prob and log_ppl
        """
        assert c_mask is not None, 'context must be supplied for this decoder.'
        assert context.ndim == 3, 'context must have 3 dimentions.'
        # context: (nb_samples, max_len, contxt_dim)

        X, X_mask, Y, Y_mask, Count = self.prepare_xy(target, context)

        # input drop-out if any.
        if self.dropout > 0:
            X     = self.D(X, train=train)

        # Initial state of RNN
        Init_h  = self.Initializer(context[:, 0, :])  # time 0 of each sequence embedding
        X       = X.dimshuffle((1, 0, 2)) # shuffle to [max_len_target, nb_sample, 2*enc_hidden_dim]
        X_mask  = X_mask.dimshuffle((1, 0)) # shuffle to [max_len_target, nb_sample]

        def _recurrence(x, x_mask, prev_h, cc, cm):
            # compute the attention
            prob  = self.attention_reader(prev_h, cc, Smask=cm)
            # get the context vector after attention by context * atten_prob
            c     = T.sum(cc * prob[:, :, None], axis=1)
            # get the RNN output vector of this step
            x_out = self.RNN(x, mask=x_mask, C=c, init_h=prev_h, one_step=True)
            # return RNN output, attention prob and attentioned context
            #    x_out is used as the prev_h of next iteration
            return x_out, prob, c

        outputs, _ = theano.scan(
            _recurrence,
            sequences=[X, X_mask],
            outputs_info=[Init_h, None, None],
            non_sequences=[context, c_mask],
            name='decoder_scan'
        )
        X_out, Probs, Ctx = [z.dimshuffle((1, 0, 2)) for z in outputs] #shuffle to [nb_sample, max_len_target, dec_hidden_dim]
        # return to normal size.
        X       = X.dimshuffle((1, 0, 2)) # shuffle back to [nb_sample, max_len_target, 2*enc_hidden_dim]
        X_mask  = X_mask.dimshuffle((1, 0)) # shuffle back to [nb_sample, max_len_target]

        # Readout
        readin  = [X_out] # RNN output at each time t
        # a linear activation layer, take input size=dec_hidden_dim, output size=dec_voc_size
        #   just return W*X+b [nb_sample, max_len_target, dec_voc_size]
        readout = self.hidden_readout(X_out)
        if self.dropout > 0:
            readout = self.D(readout, train=train)

        # don't know what's this
        #   take another linear layer context_readout, return [nb_sample, 1]
        if self.config['context_predict']:
            readin  += [Ctx]
            readout += self.context_readout(Ctx)

        # don't know what's this
        #   another linear layer prev_word_readout, return [nb_sample, 1]
        if self.config['bigram_predict']:
            readin  += [X]
            readout += self.prev_word_readout(X)

        # only have non-linear for maxout, so not work here
        for l in self.output_nonlinear:
            readout = l(readout)

        if self.copynet:
            readin  = T.concatenate(readin, axis=-1)
            key     = self.Os(readin)

            # (nb_samples, max_len_T, embed_size) :: key
            # (nb_samples, max_len_S, embed_size) :: context
            Eng     = T.sum(key[:, :, None, :] * context[:, None, :, :], axis=-1)
            # (nb_samples, max_len_T, max_len_S)  :: Eng
            EngSum  = logSumExp(Eng, axis=2, mask=c_mask[:, None, :], c=readout)
            prob_dist = T.concatenate([T.exp(readout - EngSum), T.exp(Eng - EngSum) * c_mask[:, None, :]], axis=-1)
        else:
            # output after a simple softmax, return a tensor size in [nb_samples, max_len, vocab_size], indicates the probabilities of next (predicted) word
            prob_dist = self.output(readout)  # (nb_samples, max_len, vocab_size)

        '''
        Compute Cross-entropy!
        1. _grab_prob(prob_dist, target), predicted probabilities of each target term, shape=[nb_sample, max_len_target]
              prob_dist is [nb_samples, max_len_source, vocab_size], target is [nb_sample, max_len_target]
        2. T.log(self._grab_prob(prob_dist, target)) * X_mask,
            to remove the probabilities of padding terms (index=0)
        3. sum up the log(predicted probability), to get the cross-entropy loss of target sequence (must be less than 0, add a minus to make it possitive, we want the minimize this value to 0)
        4. return a tensor in [nb_samples,1], overall loss of each sample
        [Important] Add value clipping in case of log(0)=-inf. 2016/12/11
        '''
        log_prob = T.sum(T.log(T.clip(self._grab_prob(prob_dist, target), 1e-10, 1.0)) * X_mask, axis=1)
        #  Count is number of terms in each targets
        log_ppl  = log_prob / Count

        if return_count:
            return log_prob, Count
        else:
            return log_prob, log_ppl


    def build_representer(self,
                      target,
                      context, c_mask,
                      train=True):
        """
        Very similar to build_decoder, but instead of return cross-entropy of generating target sequences,
            we return the probability of generating target sequences (similar to _step_sample)
        :param target:              index vector of target   [nb_sample, max_len]
        :param context:             encoding of source text, [nb_sample, max_len, 2*enc_hidden_dim]
        :param last_words:          the index of the last word in target

        :return prob_dist:          probability of generating target sequences, size=TODO
        :return final_state:        decode vector of generating target sequences, size=TODO
        """
        assert context.ndim == 3, 'context must have 3 dimentions.'
        # context: (nb_samples, max_len, contxt_dim)

        # prepare the inputs
        #      X                :  embedding of targets, padded first word as 0 [nb_sample, max_len, enc_embedd_dim]
        #      X_mask           :  mask of targets, padded first word as 0      [nb_sample, max_len]
        #      Y,Y_mask,Count   :  not used
        X, X_mask, _, _, _ = self.prepare_xy(target, context)

        # input drop-out if any.
        if self.dropout > 0:
            X     = self.D(X, train=train)

        # Initial state of RNN
        Init_h  = self.Initializer(context[:, 0, :])  # time 0 of each sequence embedding
        X       = X.dimshuffle((1, 0, 2))   # shuffle to [max_len_target, nb_sample, enc_hidden_dim]
        X_mask  = X_mask.dimshuffle((1, 0)) # shuffle to [max_len_target, nb_sample]

        def _recurrence(x, x_mask, prev_h, source_context, cm):
            '''

            :param x:       word embedding of current target word[nb_sample, enc_hidden_dim]
            :param x_mask:  mask of target [nb_sample]
            :param prev_h:  hidden state of previous time t-1
            :param source_context:      encoding vector of source
            :param cm:      mask of source
            :return: x_out: decoding vector after time t
            :return: prob:  attention probability
            :return: c:     context vector after attention
            '''
            # compute the attention
            attention_prob  = self.attention_reader(prev_h, source_context, Smask=cm)
            # get the context vector after attention by context * atten_prob
            c     = T.sum(source_context * attention_prob[:, :, None], axis=1)
            # get the RNN output vector of this step
            x_out = self.RNN(x, mask=x_mask, C=c, init_h=prev_h, one_step=True)
            # return RNN output, attention prob and attentioned context
            #    x_out is used as the prev_h of next iteration
            return x_out, attention_prob, c

        # return the outputs of _recurrence, update is ignored (the _)
        outputs, _ = theano.scan(
            _recurrence,
            sequences=[X, X_mask],
            outputs_info=[Init_h, None, None],
            non_sequences=[context, c_mask],
            name='decoder_scan'
        )

        # X_out is the decoding output [nb_sample, max_len_target, dec_hidden_dim]
        X_out, Probs, Ctx = [z.dimshuffle((1, 0, 2)) for z in outputs] # shuffle to [nb_sample, max_len_target, dec_hidden_dim]

        # we are only interested in the states at final, reshape to [nb_sample, dec_hidden_dim]
        final_state = X_out[ :, -1, :]
        Ctx         = Ctx[ :, -1, :]

        # return to normal size.
        X       = X.dimshuffle((1, 0, 2)) # shuffle back to [nb_sample, max_len_target, 2*enc_hidden_dim]
        X       = X[ :, -1, :]
        X_mask  = X_mask.dimshuffle((1, 0)) # shuffle back to [nb_sample, max_len_target]

        # Readout
        readin  = [final_state] # RNN output at each time t
        # a linear activation layer, take input size=[nb_sample, max_len_target, dec_hidden_dim], output size=[nb_sample, max_len_target, dec_voc_size]
        #   just return W*X+b [nb_sample, max_len_target, dec_voc_size]
        readout = self.hidden_readout(final_state)

        if self.dropout > 0:
            readout = self.D(readout, train=train)

        # don't know what's this
        #   take another linear layer context_readout, return [nb_sample, 1]
        if self.config['context_predict']:
            readin  += [Ctx]
            readout += self.context_readout(Ctx)

        # don't know what's this
        #   another linear layer prev_word_readout, return [nb_sample, 1]
        if self.config['bigram_predict']:
            readin  += [X]
            readout += self.prev_word_readout(X)

        # only have non-linear for maxout, so not work here
        for l in self.output_nonlinear:
            readout = l(readout)

        if self.copynet:
            readin  = T.concatenate(readin, axis=-1)
            key     = self.Os(readin)

            # (nb_samples, max_len_T, embed_size) :: key
            # (nb_samples, max_len_S, embed_size) :: context
            Eng     = T.sum(key[:, :, None, :] * context[:, None, :, :], axis=-1)
            # (nb_samples, max_len_T, max_len_S)  :: Eng
            EngSum  = logSumExp(Eng, axis=2, mask=c_mask[:, None, :], c=readout)
            prob_dist = T.concatenate([T.exp(readout - EngSum), T.exp(Eng - EngSum) * c_mask[:, None, :]], axis=-1)
        else:
            # output after a simple softmax, return a tensor size in [nb_samples, max_len, vocab_size], indicates the probabilities of next (predicted) word
            prob_dist = self.output(readout)  # (nb_samples, vocab_size)

        return prob_dist, final_state


    def _step_sample(self, prev_word, prev_stat, context, c_mask):
        """
        Sample one step
        """
        assert c_mask is not None, 'we need the source mask.'
        # word embedding (note that for the first word, embedding should be all zero)
        if self.config['use_input']:
            if not self.copynet:
                X = T.switch(
                    prev_word[:, None] < 0,
                    alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim']),
                    self.Embed(prev_word)
                )
            else:
                X = T.switch(
                    prev_word[:, None] < 0,
                    alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim']),
                    self.Embed(prev_word, context=self.Is(context))
                )
        else:
            X = alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim'])

        if self.dropout > 0:
            X = self.D(X, train=False)

        # apply one step of RNN
        Probs  = self.attention_reader(prev_stat, context, c_mask)
        cxt    = T.sum(context * Probs[:, :, None], axis=1)
        X_proj = self.RNN(X, C=cxt, init_h=prev_stat, one_step=True)
        next_stat = X_proj

        # compute the readout probability distribution and sample it
        # here the readout is a matrix, different from the learner.
        readout = self.hidden_readout(next_stat)
        readin  = [next_stat]
        if self.dropout > 0:
            readout = self.D(readout, train=False)

        if self.config['context_predict']:
            readout += self.context_readout(cxt)
            readin  += [cxt]

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)
            readin  += [X]

        for l in self.output_nonlinear:
            readout = l(readout)

        if self.copynet:
            readin  = T.concatenate(readin, axis=-1)
            key     = self.Os(readin)

            # (nb_samples, embed_size) :: key
            # (nb_samples, max_len_S, embed_size) :: context
            Eng     = T.sum(key[:, None, :] * context[:, :, :], axis=-1)
            # (nb_samples, max_len_S)  :: Eng
            EngSum  = logSumExp(Eng, axis=-1, mask=c_mask, c=readout)
            next_prob = T.concatenate([T.exp(readout - EngSum), T.exp(Eng - EngSum) * c_mask], axis=-1)
        else:
            next_prob = self.output(readout)  # (nb_samples, max_len, vocab_size)

        next_sample = self.rng.multinomial(pvals=next_prob).argmax(1)
        return next_prob, next_sample, next_stat

    def build_sampler(self):
        """
        Build a sampler which only steps once.
        Typically it only works for one word a time?
        """
        logger.info("build sampler ...")
        if self.config['sample_stoch'] and self.config['sample_argmax']:
            logger.info("use argmax search!")
        elif self.config['sample_stoch'] and (not self.config['sample_argmax']):
            logger.info("use stochastic sampling!")
        elif self.config['sample_beam'] > 1:
            logger.info("use beam search! (beam_size={})".format(self.config['sample_beam']))

        # initial state of our Decoder.
        context = T.tensor3()  # theano variable.
        c_mask  = T.matrix()   # mask of the input sentence.

        init_h = self.Initializer(context[:, 0, :])
        logger.info('compile the function: get_init_state')
        self.get_init_state \
            = theano.function([context], init_h, name='get_init_state')
        logger.info('done.')

        # word sampler: 1 x 1
        prev_word = T.vector('prev_word', dtype='int64')
        prev_stat = T.matrix('prev_state', dtype='float32')

        next_prob, next_sample, next_stat \
            = self._step_sample(prev_word, prev_stat, context, c_mask)

        # next word probability
        logger.info('compile the function: sample_next')
        inputs = [prev_word, prev_stat, context, c_mask]
        outputs = [next_prob, next_sample, next_stat]

        self.sample_next = theano.function(inputs, outputs, name='sample_next', allow_input_downcast=True)
        logger.info('done')
        pass

    def get_sample(self, encoding, c_mask, inputs,
                   k=1, maxlen=30, stochastic=True, argmax=False, fixlen=False,
                   type='extractive'):
        '''
        Generate samples, either with stochastic sampling or beam-search!
        both inputs and context contain multiple sentences, so could this function generate sequence with regard to each input spontaneously?
        :param inputs: the source text, used for extraction
        :param encoding: the encoding of input sequence on each word, shape=[len(sent),2*D], 2*D is due to bidirectional
        :param c_mask: whether x in input is not zero (is padding)
        :param k: config['sample_beam']
        :param maxlen: config['max_len']
        :param stochastic: config['sample_stoch']
        :param argmax: config['sample_argmax']
        :param fixlen:
        :return:
        '''
        # beam size
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling!!'

        # fix length cannot use beam search
        # if fixlen:
        #     assert k == 1

        # prepare for searching
        sample = []
        score  = []
        state  = []

        # if stochastic:
        #     score = 0

        live_k = 1
        dead_k = 0

        # initial prediction pool
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype(theano.config.floatX)
        hyp_states = []

        # get initial state of decoder RNN with context, size = 1*D
        previous_state = self.get_init_state(encoding)
        # indicator for the first target word (bos target). Why it's [-1]?
        previous_word = -1 * np.ones((1,)).astype('int32')

        # if aim is extractive, then set the initial beam size to be voc_size
        if type == 'extractive':
            input = inputs[0]
            input_set = set(input)
            sequence_set = set()

            for i in range(len(input)): # loop over start
                for j in range(1, maxlen): # loop over length
                    if i+j > len(input)-1:
                        break
                    hash_token = [str(s) for s in input[i:i+j]]
                    sequence_set.add('-'.join(hash_token))
            logger.info("Possible n-grams: %d" % len(sequence_set))

        # Start searching!
        for ii in xrange(maxlen):
            # predict next_word
            # make live_k copies of context and c_mask, to predict next word at once
            encoding_copies    = np.tile(encoding, [live_k, 1, 1])
            c_mask_copies      = np.tile(c_mask, [live_k, 1])
            '''
            based on the live_k alive prediction, predict the next word of them at a time.
            Inputs:
                previous_word:      a list of index of last word, size=[live_k]
                previous_state:     decoding vector of time t-1, size= [live_k, dec_hidden_dim]
                encoding_copies:    encoding of source, size=[live_k, len_source, 2*enc_hidden_dim]
                c_mask_copies:      mask of source ,    size=[live_k, len_source]
            Return live_k groups of results (next_prob, next_word, next_state)
                next_prob is [live_k, voc_size], contains the probabilities of predicted next word.
                next_word is a list of [live_k] given by = self.rng.multinomial(pvals=next_prob).argmax(1), not useful for beam-search.
                next_state is the current hidden state of decoder, size=[live_k, dec_hidden_dim]
            '''
            next_prob, next_word, next_state \
                = self.sample_next(previous_word, previous_state, encoding_copies, c_mask_copies)

            if stochastic:
                # using stochastic sampling (or greedy sampling.)
                if argmax:
                    nw = next_prob[0].argmax()
                    next_word[0] = nw
                else:
                    nw = next_word[0]

                # sample.append(nw)
                # score += next_prob[0, nw]

                if (not fixlen) and (nw == 0):  # sample reached the end
                    break

            else:
                # using beam-search
                # we can only compute in a flatten way!
                cand_scores = hyp_scores[:, None] - np.log(next_prob + 1e-10) # the smaller the better
                cand_flat = cand_scores.flatten() # transform the k*V into a list of [1*kV]
                # get the index of highest words for each beam
                ranks_flat = cand_flat.argsort()[:(k - dead_k)] # get the (global) top k prediction words

                # fetch the best results. Get the index of best predictions. trans_index is the index of its previous word, word_index is the index of prediction
                voc_size                = next_prob.shape[1]
                previous_sequence_index = ranks_flat / voc_size
                next_word_index         = ranks_flat % voc_size
                costs                   = cand_flat[ranks_flat]

                # get the new hyp samples
                new_hyp_samples = []
                new_hyp_scores = np.zeros(k - dead_k).astype(theano.config.floatX)
                new_hyp_states = []
                # enumerate (last word, predicted word), store corresponding: 1. new_hyp_samples: current sequence; 2.new_hyp_scores: current score (probability); 3. new_hyp_states: the hidden state
                for idx, [ti, wi] in enumerate(zip(previous_sequence_index, next_word_index)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                # check all the predictions
                for idx in xrange(len(new_hyp_samples)):
                    if (new_hyp_samples[idx][-1] == 0) and (not fixlen): # bug??? why new_hyp_states[idx][-1] == 0? I think it should be new_hyp_samples[idx][-1] == 0
                        # if the predicted words is <eol>(reach the end), add to final list
                        # sample.append(new_hyp_samples[idx])
                        # score.append(new_hyp_scores[idx])

                        # disable the dead_k to extend the prediction
                        # dead_k += 1
                        sample.append(new_hyp_samples[idx])
                        score.append(new_hyp_scores[idx])
                        state.append(new_hyp_states[idx])
                    else:
                        # not end, check whether current new_hyp_samples[idx] is in original text,
                        # if yes, add the queue for predicting next round
                        # if no, discard
                        # limit predictions must appear in text
                        if type == 'extractive':
                            '''
                            only predict the subsequences that appear in the original text
                            actually not all the n-grams will be included
                            as only after predicting a <eol> , this prediction will be put into final results
                            '''
                            if new_hyp_samples[idx][-1] not in input_set:
                                continue

                            # TODO something wrong here
                            if '-'.join([str(s) for s in new_hyp_samples[idx]]) not in sequence_set:
                                continue

                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])

                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                # ending condition
                if new_live_k < 1:
                    break
                # if dead_k >= k:
                #     break

                '''
                set the predicted word and hidden_state as the next_word and next_state
                '''
                previous_word = np.array([w[-1] for w in hyp_samples])
                previous_state = np.array(hyp_states)

            logger.info('\t Depth=%d, get %d outputs' % (ii, len(sample)))

        # end.
        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    score.append(hyp_scores[idx])
                    state.append(hyp_states[idx])

        # sort the result
        result = zip(sample, score, state)
        sorted_result = sorted(result, key=lambda entry: entry[1], reverse=False)
        sample, score, state = zip(*sorted_result)
        return sample, score, state



class FnnDecoder(Model):
    def __init__(self, config, rng, prefix='fnndec'):
        """
        mode = RNN: use a RNN Decoder
        """
        super(FnnDecoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix
        self.name = prefix

        """
        Create Dense Predictor.
        """

        self.Tr = Dense(self.config['dec_contxt_dim'],
                             self.config['dec_hidden_dim'],
                             activation='maxout2',
                             name='{}_Tr'.format(prefix))
        self._add(self.Tr)

        self.Pr = Dense(self.config['dec_hidden_dim'] / 2,
                             self.config['dec_voc_size'],
                             activation='softmax',
                             name='{}_Pr'.format(prefix))
        self._add(self.Pr)
        logger.info("FF decoder ok.")

    @staticmethod
    def _grab_prob(probs, X):
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    def build_decoder(self, target, context):
        """
        Build the Decoder Computational Graph
        """
        prob_dist = self.Pr(self.Tr(context[:, None, :]))
        log_prob  = T.sum(T.log(self._grab_prob(prob_dist, target)), axis=1)
        return log_prob

    def build_sampler(self):
        context   = T.matrix()
        prob_dist = self.Pr(self.Tr(context))
        next_sample = self.rng.multinomial(pvals=prob_dist).argmax(1)
        self.sample_next = theano.function([context], [prob_dist, next_sample], name='sample_next_{}'.format(self.prefix))
        logger.info('done')

    def get_sample(self, context, argmax=True):

        prob, sample = self.sample_next(context)
        if argmax:
            return prob[0].argmax()
        else:
            return sample[0]


########################################################################################################################
# Encoder-Decoder Models ::::
#
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
        logger.info("build the RNN-decoder")
        self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)

        # registration:
        self._add(self.decoder)

        # objectives and optimizers
        self.optimizer = optimizers.get('adadelta')

        # saved the initial memories
        if self.config['mode'] == 'NTM':
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
                                      name='train_fun',
                                      allow_input_downcast=True
                                      )

        logger.info("pre-training functions compile done.")

        # add monitoring:
        self.monitor['context'] = context
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)

    @abstractmethod
    def compile_train_CE(self):
        pass

    def compile_sample(self):
        # context vectors (as)
        self.decoder.build_sampler()
        logger.info("display functions compile done.")

    @abstractmethod
    def compile_inference(self):
        pass

    def default_context(self):
        if self.config['mode'] == 'RNN':
            return np.zeros(shape=(1, self.config['dec_contxt_dim']), dtype=theano.config.floatX)
        elif self.config['mode'] == 'NTM':
            memory = self.memory.get_value()
            memory = memory.reshape((1, memory.shape[0], memory.shape[1]))
            return memory

    def generate_(self, context=None, max_len=None, mode='display'):
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


class AutoEncoder(RNNLM):
    """
    Regular Auto-Encoder: RNN Encoder/Decoder
    """

    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name = 'vae'

    def build_(self):
        logger.info("build the RNN auto-encoder")
        self.encoder = Encoder(self.config, self.rng, prefix='enc')
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', embed=self.encoder.Embed)
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec')

        """
        Build the Transformation
        """
        if self.config['nonlinear_A']:
            self.action_trans = Dense(
                self.config['enc_hidden_dim'],
                self.config['action_dim'],
                activation='tanh',
                name='action_transform'
            )
        else:
            assert self.config['enc_hidden_dim'] == self.config['action_dim'], \
                    'hidden dimension must match action dimension'
            self.action_trans = Identity(name='action_transform')

        if self.config['nonlinear_B']:
            self.context_trans = Dense(
                self.config['action_dim'],
                self.config['dec_contxt_dim'],
                activation='tanh',
                name='context_transform'
            )
        else:
            assert self.config['dec_contxt_dim'] == self.config['action_dim'], \
                    'action dimension must match context dimension'
            self.context_trans = Identity(name='context_transform')

        # registration
        self._add(self.action_trans)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # objectives and optimizers
        self.optimizer = optimizers.get(self.config['optimizer'], kwargs={'lr': self.config['lr']})

        logger.info("create Helmholtz RECURRENT neural network. ok")

    def compile_train(self, mode='train'):
        # questions (theano variables)
        inputs  = T.imatrix()  # padded input word sequence (for training)
        context = alloc_zeros_matrix(inputs.shape[0], self.config['dec_contxt_dim'])
        assert context.ndim == 2

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

        if mode == 'display' or mode == 'all':
            """
            build the sampler function here <:::>
            """
            # context vectors (as)
            self.decoder.build_sampler()
            logger.info("display functions compile done.")

        # add monitoring:
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)


class NRM(Model):
    """
    Neural Responding Machine
    A Encoder-Decoder based responding model.
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation',
                 use_attention=False,
                 copynet=False,
                 identity=False):
        super(NRM, self).__init__()

        self.config   = config
        self.n_rng    = n_rng  # numpy random stream
        self.rng      = rng  # Theano random stream
        self.mode     = mode
        self.name     = 'nrm'
        self.attend   = use_attention
        self.copynet  = copynet
        self.identity = identity

    def build_(self):
        logger.info("build the Neural Responding Machine")

        # encoder-decoder:: <<==>>
        self.encoder = Encoder(self.config, self.rng, prefix='enc', mode=self.mode)
        if not self.attend:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)
        else:
            self.decoder = DecoderAtt(self.config, self.rng, prefix='dec', mode=self.mode,
                                      copynet=self.copynet, identity=self.identity)

        self._add(self.encoder)
        self._add(self.decoder)

        # objectives and optimizers
        # self.optimizer = optimizers.get(self.config['optimizer'])
        assert self.config['optimizer'] == 'adam'
        self.optimizer = optimizers.get(self.config['optimizer'],
                                        kwargs=dict(rng=self.rng,
                                                    save=False,
                                                    clipnorm = self.config['clipnorm']))
        logger.info("build ok.")

    def compile_(self, mode='all', contrastive=False):
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

    def compile_train(self):

        # questions (theano variables)
        inputs  = T.imatrix()  # padded input word sequence (for training)
        target  = T.imatrix()  # padded target word sequence (for training)

        # encoding & decoding
        if not self.attend:
            code               = self.encoder.build_encoder(inputs, None)
            logPxz, logPPL     = self.decoder.build_decoder(target, code)
        else:
            # encode text by encoder, return encoded vector at each time (code) and mask showing non-zero elements
            code, _, c_mask, _ = self.encoder.build_encoder(inputs, None, return_sequence=True, return_embed=True)
            # feed target(index vector of target), code(encoding of source text), c_mask (mask of source text) into decoder, get objective value
            #    logPxz,logPPL are tensors in [nb_samples,1], cross-entropy and Perplexity of each sample
            logPxz, logPPL     = self.decoder.build_decoder(target, code, c_mask)
            prob, stat         = self.decoder.build_representer(target, code, c_mask)


        # responding loss
        loss_rec = -logPxz # get the mean of cross-entropy of this batch
        loss_ppl = T.exp(-logPPL)
        loss     = T.mean(loss_rec)

        updates  = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs, target]

        self.train_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_fun'
                                      # , mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                      # , mode='DebugMode'
                                      )

        self.train_guard = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_nanguard_fun',
                                      mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        self.validate_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      name='validate_fun',
                                      allow_input_downcast=True)

        self.represent_ = theano.function(train_inputs,
                                      [prob, stat],
                                      name='represent_fun',
                                      allow_input_downcast=True
                                      )


        logger.info("training functions compile done.")

        # # add monitoring:
        # self.monitor['context'] = context
        # self._monitoring()
        #
        # # compiling monitoring
        # self.compile_monitoring(train_inputs)

    def compile_sample(self):
        if not self.attend:
            self.encoder.compile_encoder(with_context=False)
        else:
            self.encoder.compile_encoder(with_context=False, return_sequence=True, return_embed=True)

        self.decoder.build_sampler()
        logger.info("sampling functions compile done.")

    def compile_inference(self):
        pass

    def generate_(self, inputs, mode='display', return_all=False):
        '''
        Generate output sequence with regards to the given input sequences
        :param inputs:
        :param mode:
        :param return_all:
        :return:
        '''
        # assert self.config['sample_stoch'], 'RNNLM sampling must be stochastic'
        # assert not self.config['sample_argmax'], 'RNNLM sampling cannot use argmax'

        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'],
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None)

        if not self.attend:
            context = self.encoder.encode(inputs)
            sample, score = self.decoder.get_sample(context, **args)
        else:
            # context: input sentence embedding, the last state of sequence
            # c_mask:  whether x in input is not zero (is padding)
            context, _, c_mask, _ = self.encoder.encode(inputs)
            sample, score = self.decoder.get_sample(context, c_mask, **args)

        if return_all:
            return sample, score

        if not args['stochastic']:
            score = score / np.array([len(s) for s in sample])
            sample = sample[score.argmin()]
            score = score.min()
        else:
            score /= float(len(sample))

        return sample, np.exp(score)

    def generate_multiple(self, inputs, mode='display', return_all=True, all_ngram=True, generate_ngram=True):
        '''
        Generate output sequence
        '''
        # assert self.config['sample_stoch'], 'RNNLM sampling must be stochastic'
        # assert not self.config['sample_argmax'], 'RNNLM sampling cannot use argmax'

        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'],
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None,
                    type=self.config['predict_type'])

        if not self.attend:
            # get the encoding of the inputs
            context = self.encoder.encode(inputs)
            # generate outputs
            sample, score, _ = self.decoder.get_sample(context, inputs, **args)
        else:
            # context: input sentence embedding
            # c_mask:  whether x in input is not zero (is padding)
            context, _, c_mask, _ = self.encoder.encode(inputs)
            sample, score, _ = self.decoder.get_sample(context, c_mask, inputs, **args)

        if return_all:
            return sample, score

        if not args['stochastic']:
            score = score / np.array([len(s) for s in sample])
            sample = sample[score.argmin()]
            score = score.min()
        else:
            score /= float(len(sample))

        return sample, np.exp(score)

    # def evaluate_(self, inputs, outputs, idx2word,
    #               origin=None, idx2word_o=None):
    #     '''
    #     This function doesn't support the <unk>, don't use this if voc_size is set
    #     :param inputs:
    #     :param outputs:
    #     :param idx2word:
    #     :param origin:
    #     :param idx2word_o:
    #     :return:
    #     '''
    #
    #     def cut_zero(sample, idx2word, idx2word_o):
    #         Lmax = len(idx2word)
    #         if not self.copynet:
    #             if 0 not in sample:
    #                 return [idx2word[w] for w in sample]
    #             return [idx2word[w] for w in sample[:sample.index(0)]]
    #         else:
    #             if 0 not in sample:
    #                 if origin is None:
    #                     return [idx2word[w] if w < Lmax else idx2word[inputs[w - Lmax]]
    #                             for w in sample]
    #                 else:
    #                     return [idx2word[w] if w < Lmax else idx2word_o[origin[w - Lmax]]
    #                             for w in sample]
    #             if origin is None:
    #                 return [idx2word[w] if w < Lmax else idx2word[inputs[w - Lmax]]
    #                         for w in sample[:sample.index(0)]]
    #             else:
    #                 return [idx2word[w] if w < Lmax else idx2word_o[origin[w - Lmax]]
    #                         for w in sample[:sample.index(0)]]
    #
    #     result, _ = self.generate_(inputs[None, :])
    #
    #     if origin is not None:
    #         logger.info( '[ORIGIN]: {}'.format(' '.join(cut_zero(origin.tolist(), idx2word_o, idx2word_o))))
    #     logger.info('[DECODE]: {}'.format(' '.join(cut_zero(result, idx2word, idx2word_o))))
    #     logger.info('[SOURCE]: {}'.format(' '.join(cut_zero(inputs.tolist(),  idx2word, idx2word_o))))
    #     logger.info('[TARGET]: {}'.format(' '.join(cut_zero(outputs.tolist(), idx2word, idx2word_o))))
    #
    #     return True
    #
    def evaluate_(self, inputs, outputs, idx2word, inputs_unk=None):

        def cut_zero(sample, idx2word, Lmax=None):
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if 0 not in sample:
                return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]
            return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

        if inputs_unk is None:
            result, _ = self.generate_(inputs[None, :])
        else:
            result, _ = self.generate_(inputs_unk[None, :])

        a = '[SOURCE]: {}'.format(' '.join(cut_zero(inputs.tolist(),  idx2word)))
        b = '[TARGET]: {}'.format(' '.join(cut_zero(outputs.tolist(), idx2word)))
        c = '[DECODE]: {}'.format(' '.join(cut_zero(result, idx2word)))
        print(a)
        if inputs_unk is not None:
            k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
            print(k)
            a += k
        print(b)
        print(c)
        a += b + c
        return a

    def evaluate_multiple(self, inputs, outputs,
                            original_input, original_outputs,
                            samples, scores, idx2word):
        '''
        inputs_unk is same as inputs except for filtered out all the low-freq words to 1 (<unk>)
        return the top few keywords, number is set in config
        :param: original_input, same as inputs, the vector of one input sentence
        :param: original_outputs, vectors of corresponding multiple outputs (e.g. keyphrases)
        :return:
        '''

        def cut_zero(sample, idx2word, Lmax=None):
            sample = list(sample)
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if 0 not in sample:
                return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]
            # return the string before 0 (<eol>)
            return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

        # Generate keyphrases
        # if inputs_unk is None:
        #     samples, scores = self.generate_multiple(inputs[None, :], return_all=True)
        # else:
        #     samples, scores = self.generate_multiple(inputs_unk[None, :], return_all=True)

        stemmer = PorterStemmer()
        # Evaluation part
        outs = []
        metrics = []

        # load stopword
        with open(self.config['path']+'/dataset/stopword/stopword_en.txt') as stopword_file:
            stopword_set = set([stemmer.stem(w.strip()) for w in stopword_file])

        for input_sentence, target_list, predict_list, score_list in zip(inputs, original_outputs, samples, scores):
            '''
            enumerate each document, process target/predict/score and measure via p/r/f1
            '''
            target_outputs  = []
            predict_outputs = []
            predict_scores  = []
            predict_set     = set()
            correctly_matched = np.asarray([0]*max(len(target_list), len(predict_list)), dtype='int32')

            # stem the original input
            stemmed_input = [stemmer.stem(w) for w in cut_zero(input_sentence, idx2word)]

            # convert target index into string
            for target in target_list:
                target = cut_zero(target, idx2word)
                target = [stemmer.stem(w) for w in target]

                keep = True
                # whether do filtering on groundtruth phrases. if config['target_filter']==None, do nothing
                if self.config['target_filter']:
                    match = None
                    for i in range(len(stemmed_input) - len(target) + 1):
                        match = None
                        j = 0
                        for j in range(len(target)):
                            if target[j] != stemmed_input[i + j]:
                                match = False
                                break
                        if j == len(target)-1 and match == None:
                            match = True
                            break

                    if match == True:
                        # if match and 'appear-only', keep this phrase
                        if self.config['target_filter'] == 'appear-only':
                            keep = keep and True
                        elif self.config['target_filter'] == 'non-appear-only':
                            keep = keep and False
                    elif match == False:
                        # if not match and 'appear-only', discard this phrase
                        if self.config['target_filter'] == 'appear-only':
                            keep = keep and False
                        # if not match and 'non-appear-only', keep this phrase
                        elif self.config['target_filter'] == 'non-appear-only':
                            keep = keep and True

                if not keep:
                    continue

                target_outputs.append(target)

            # convert predict index into string
            for id, (predict, score) in enumerate(zip(predict_list, score_list)):
                predict = cut_zero(predict, idx2word)
                predict = [stemmer.stem(w) for w in predict]

                # filter some not good ones
                keep = True
                if len(predict) == 0:
                    keep = False
                number_digit = 0
                for w in predict:
                    if w.strip() == '<unk>':
                        keep = False
                    if w.strip() == '<digit>':
                        number_digit += 1

                if len(predict) >= 1 and (predict[0] in stopword_set or predict[-1] in stopword_set):
                    keep = False

                if len(predict) <= 1:
                    keep = False

                # whether do filtering on predicted phrases. if config['predict_filter']==None, do nothing
                if self.config['predict_filter']:
                    match = None
                    for i in range(len(stemmed_input) - len(predict) + 1):
                        match = None
                        j = 0
                        for j in range(len(predict)):
                            if predict[j] != stemmed_input[i + j]:
                                match = False
                                break
                        if j == len(predict)-1 and match == None:
                            match = True
                            break

                    if match == True:
                        # if match and 'appear-only', keep this phrase
                        if self.config['predict_filter'] == 'appear-only':
                            keep = keep and True
                        elif self.config['predict_filter'] == 'non-appear-only':
                            keep = keep and False
                    elif match == False:
                        # if not match and 'appear-only', discard this phrase
                        if self.config['predict_filter'] == 'appear-only':
                            keep = keep and False
                        # if not match and 'non-appear-only', keep this phrase
                        elif self.config['predict_filter'] == 'non-appear-only':
                            keep = keep and True

                key = '-'.join(predict)
                # remove this phrase and its score from list
                if not keep or number_digit == len(predict) or key in predict_set:
                    continue

                predict_outputs.append(predict)
                predict_scores.append(score)
                predict_set.add(key)

                # check whether correct
                for target in target_outputs:
                    if len(target)==len(predict):
                        flag = True
                        for i,w in enumerate(predict):
                            if predict[i]!=target[i]:
                                flag = False
                        if flag:
                            correctly_matched[len(predict_outputs) - 1] = 1
                        # print('%s correct!!!' % predict)

            predict_outputs = np.asarray(predict_outputs)
            predict_scores = np.asarray(predict_scores)
            # normalize the score?
            if self.config['normalize_score']:
                predict_scores = np.asarray([math.log(math.exp(score)/len(predict)) for predict, score in zip(predict_outputs, predict_scores)])
                score_list_index = np.argsort(predict_scores)
                predict_outputs = predict_outputs[score_list_index]
                predict_scores = predict_scores[score_list_index]
                correctly_matched = correctly_matched[score_list_index]

            metric_dict = {}

            for number_to_predict in [5,10,15]:
                metric_dict['p@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict]))/float(number_to_predict)

                if len(target_outputs) != 0:
                    metric_dict['r@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict]))/float(len(target_outputs))
                else:
                    metric_dict['r@%d' % number_to_predict] = 0

                if metric_dict['p@%d' % number_to_predict]+metric_dict['r@%d' % number_to_predict] != 0:
                    metric_dict['f1@%d' % number_to_predict]= 2*metric_dict['p@%d' % number_to_predict]*metric_dict['r@%d' % number_to_predict]/float(metric_dict['p@%d' % number_to_predict]+metric_dict['r@%d' % number_to_predict])
                else:
                    metric_dict['f1@%d' % number_to_predict] = 0

                metric_dict['valid_target_number']  = len(target_outputs)
                metric_dict['target_number']  = len(target_list)
                metric_dict['correct_number@%d' % number_to_predict] = sum(correctly_matched[:number_to_predict])

            metrics.append(metric_dict)

            # print(stuff)
            a = '[SOURCE]: {}\n'.format(' '.join(cut_zero(input_sentence,  idx2word)))
            logger.info(a)

            b = '[TARGET]: %d/%d targets\n\t\t' % (len(target_outputs), len(target_list))
            for id, target in enumerate(target_outputs):
                b += ' '.join(target) + '; '
            b += '\n'
            logger.info(b)
            c = '[DECODE]: %d/%d predictions' % (len(predict_outputs), len(predict_list))
            for id, (predict, score) in enumerate(zip(predict_outputs, predict_scores)):
                if correctly_matched[id]==1:
                    c += ('\n\t\t[%.3f]'% score) + ' '.join(predict) + ' [correct!]'
                    # print(('\n\t\t[%.3f]'% score) + ' '.join(predict) + ' [correct!]')
                else:
                    c += ('\n\t\t[%.3f]'% score) + ' '.join(predict)
                    # print(('\n\t\t[%.3f]'% score) + ' '.join(predict))
            c += '\n'

            # c = '[DECODE]: {}'.format(' '.join(cut_zero(phrase, idx2word)))
            # if inputs_unk is not None:
            #     k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
            #     logger.info(k)
                # a += k
            logger.info(c)
            a += b + c

            for number_to_predict in [5,10,15]:
                d = '@%d - Precision=%.4f, Recall=%.4f, F1=%.4f\n' % (number_to_predict, metric_dict['p@%d' % number_to_predict],metric_dict['r@%d' % number_to_predict],metric_dict['f1@%d' % number_to_predict])
                logger.info(d)
                a += d

            outs.append(a)

        return outs, metrics

    def analyse_(self, inputs, outputs, idx2word):
        Lmax = len(idx2word)

        def cut_zero(sample, idx2word):
            if 0 not in sample:
                return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]

            return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

        result, _ = self.generate_(inputs[None, :])
        flag   = 0
        source = '{}'.format(' '.join(cut_zero(inputs.tolist(),  idx2word)))
        target = '{}'.format(' '.join(cut_zero(outputs.tolist(), idx2word)))
        result = '{}'.format(' '.join(cut_zero(result, idx2word)))

        return target == result

    def analyse_cover(self, inputs, outputs, idx2word):
        Lmax = len(idx2word)

        def cut_zero(sample, idx2word):
            if 0 not in sample:
                return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]

            return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

        results, _ = self.generate_(inputs[None, :], return_all=True)
        flag   = 0
        source = '{}'.format(' '.join(cut_zero(inputs.tolist(),  idx2word)))
        target = '{}'.format(' '.join(cut_zero(outputs.tolist(), idx2word)))

        score  = [target == '{}'.format(' '.join(cut_zero(result, idx2word))) for result in results]
        return max(score)