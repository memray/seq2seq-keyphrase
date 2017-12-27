import logging
import time
import numpy as np
import sys
import copy
import math

import theano

import keyphrase_utils
from keyphrase.dataset import keyphrase_test_dataset
import os
import resource
# resource.setrlimit(resource.RLIMIT_STACK, (2**10,-1))
# sys.setrecursionlimit(10**7)

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


theano.config.optimizer='fast_compile'
os.environ['THEANO_FLAGS'] = 'device=cpu'

from emolga.basic import optimizers

theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from keyphrase.dataset.keyphrase_train_dataset import *
from keyphrase.config import *
from emolga.utils.generic_utils import *
from emolga.models.covc_encdec import NRM
from emolga.models.encdec import NRM as NRM0
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes

setup = setup_keyphrase_all # setup_keyphrase_all_testing

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )
    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()
    ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging


def output_stream(dataset, batch_size, size=1):
    data_stream = dataset.get_example_stream()
    data_stream = transformers.Batch(data_stream,
                                     iteration_scheme=schemes.ConstantScheme(batch_size))

    # add padding and masks to the dataset
    # Warning: in multiple output case, will raise ValueError: All dimensions except length must be equal, need padding manually
    # data_stream = transformers.Padding(data_stream, mask_sources=('source', 'target', 'target_c'))
    # data_stream = transformers.Padding(data_stream, mask_sources=('source', 'target'))
    return data_stream


def prepare_batch(batch, mask, fix_len=None):
    data = batch[mask].astype('int32')
    data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype='int32')], axis=1)

    def cut_zeros(data, fix_len=None):
        if fix_len is not None:
            return data[:, : fix_len]
        for k in range(data.shape[1] - 1, 0, -1):
            data_col = data[:, k].sum()
            if data_col > 0:
                return data[:, : k + 2]
        return data

    data = cut_zeros(data, fix_len)
    return data


def cc_martix(source, target):
    '''
    return the copy matrix, size = [nb_sample, max_len_source, max_len_target]
    '''
    cc = np.zeros((source.shape[0], target.shape[1], source.shape[1]), dtype='float32')
    for k in range(source.shape[0]): # go over each sample in source batch
        for j in range(target.shape[1]): # go over each word in target (all target have same length after padding)
            for i in range(source.shape[1]): # go over each word in source
                if (source[k, i] == target[k, j]) and (source[k, i] > 0): # if word match, set cc[k][j][i] = 1. Don't count non-word(source[k, i]=0)
                    cc[k][j][i] = 1.
    return cc

def unk_filter(data):
    '''
    only keep the top [voc_size] frequent words, replace the other as 0
    word index is in the order of from most frequent to least
    :param data:
    :return:
    '''
    if config['voc_size'] == -1:
        return copy.copy(data)
    else:
        # mask shows whether keeps each word (frequent) or not, only word_index<config['voc_size']=1, else=0
        mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
        # low frequency word will be set to 1 (index of <unk>)
        data = copy.copy(data * mask + (1 - mask))
        return data


def add_padding(data):
    shapes = [np.asarray(sample).shape for sample in data]
    lengths = [shape[0] for shape in shapes]

    # make sure there's at least one zero at last to indicate the end of sentence <eol>
    max_sequence_length = max(lengths) + 1
    rest_shape = shapes[0][1:]
    padded_batch = np.zeros(
        (len(data), max_sequence_length) + rest_shape,
        dtype='int32')
    for i, sample in enumerate(data):
        padded_batch[i, :len(sample)] = sample

    return padded_batch


def split_into_multiple_and_padding(data_s_o, data_t_o):
    data_s = []
    data_t = []
    for s, t in zip(data_s_o, data_t_o):
        for p in t:
            data_s += [s]
            data_t += [p]

    data_s = add_padding(data_s)
    data_t = add_padding(data_t)
    return data_s, data_t

def build_data(data):
    # create fuel dataset.
    dataset = datasets.IndexableDataset(indexables=OrderedDict([('source', data['source']),
                                                                ('target', data['target']),
                                                                # ('target_c', data['target_c']),
                                                                ]))
    dataset.example_iteration_scheme \
        = schemes.ShuffledExampleScheme(dataset.num_examples)
    return dataset


if __name__ == '__main__':

    # prepare logging.
    config  = setup()   # load settings.

    print('Log path: %s' % (config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'],config['timemark'])))
    logger  = init_logging(config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'],config['timemark']))

    n_rng   = np.random.RandomState(config['seed'])
    np.random.seed(config['seed'])
    rng     = RandomStreams(n_rng.randint(2 ** 30))

    logger.info('*'*20 + '  config information  ' + '*'*20)
    # print config information
    for k,v in config.items():
        logger.info("\t\t\t\t%s : %s" % (k,v))
    logger.info('*' * 50)

    # data is too large to dump into file, so has to load from raw dataset directly
    # train_set, test_set, idx2word, word2idx = keyphrase_dataset.load_data_and_dict(config['training_dataset'], config['testing_dataset'])

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = keyphrase_test_dataset.load_additional_testing_data(['inspec'], idx2word, word2idx, config, postagging=False)

    logger.info('#(training paper)=%d' % len(train_set['source']))
    logger.info('#(training keyphrase)=%d' % sum([len(t) for t in train_set['target']]))
    logger.info('#(testing paper)=%d' % sum([len(test_set['target']) for test_set in test_sets.values()]))

    logger.info('Load data done.')

    if config['voc_size'] == -1:   # not use unk
        config['enc_voc_size'] = max(list(zip(*word2idx.items()))[1]) + 1
        config['dec_voc_size'] = config['enc_voc_size']
    else:
        config['enc_voc_size'] = config['voc_size']
        config['dec_voc_size'] = config['enc_voc_size']

    predictions  = len(train_set['source'])

    logger.info('build dataset done. ' +
                'dataset size: {} ||'.format(predictions) +
                'vocabulary size = {0}/ batch size = {1}'.format(
            config['dec_voc_size'], config['batch_size']))

    # train_data        = build_data(train_set) # a fuel IndexableDataset
    train_data_plain  = list(zip(*(train_set['source'], train_set['target'])))
    train_data_source = np.array(train_set['source'])
    train_data_target = np.array(train_set['target'])

    count_has_OOV = 0
    count_all     = 0
    for phrases in train_data_target:
        for phrase in phrases:
            count_all += 1
            if np.greater(phrase, np.asarray(config['voc_size'])).any():
                count_has_OOV += 1
    print('%d / %d' % (count_has_OOV, count_all))

    # test_data_plain   = list(zip(*(test_set['source'],  test_set['target'])))

    # trunk the over-long input in testing data
    # for test_set in test_sets.values():
    #     test_set['source'] = [s if len(s)<1000 else s[:1000] for s in test_set['source']]
    test_data_plain = np.concatenate([list(zip(*(t['source'],  t['target']))) for k,t in test_sets.items()])

    print('Avg length=%d, Max length=%d' % (
    np.average([len(s[0]) for s in test_data_plain]), np.max([len(s[0]) for s in test_data_plain])))

    train_size        = len(train_data_plain)
    test_size         = len(test_data_plain)
    tr_idx            = n_rng.permutation(train_size)[:2000].tolist()
    ts_idx            = n_rng.permutation(test_size )[:2000].tolist()
    logger.info('load the data ok.')

    if config['do_train'] or config['do_predict']:
        # build the agent
        if config['copynet']:
            agent = NRM(config, n_rng, rng, mode=config['mode'],
                         use_attention=True, copynet=config['copynet'], identity=config['identity'])
        else:
            agent = NRM0(config, n_rng, rng, mode=config['mode'],
                          use_attention=True, copynet=config['copynet'], identity=config['identity'])

        agent.build_()
        agent.compile_('all')
        logger.info('compile ok.')

        # load pre-trained model to continue training
        if config['trained_model'] and os.path.exists(config['trained_model']):
            logger.info('Trained model exists, loading from %s' % config['trained_model'])
            agent.load(config['trained_model'])
            # agent.save_weight_json(config['weight_json'])

    epoch   = 0
    epochs = 10
    valid_param = {}
    valid_param['early_stop'] = False
    valid_param['valid_best_score'] = (float(sys.maxsize),float(sys.maxsize))
    valid_param['valids_not_improved'] = 0
    valid_param['patience']            = 3

    # do training?
    do_train     = config['do_train']
    # do predicting?
    do_predict     = config['do_predict']
    # do testing?
    do_evaluate     = config['do_evaluate']
    do_validate     = config['do_validate']

    if do_train:
        while epoch < epochs:
            epoch += 1
            loss  = []
            # train_batches = output_stream(train_data, config['batch_size']).get_epoch_iterator(as_dict=True)

            if valid_param['early_stop']:
                break

            logger.info('\nEpoch = {} -> Training Set Learning...'.format(epoch))
            progbar = Progbar(train_size / config['batch_size'], logger)

            # number of minibatches
            num_batches = int(float(len(train_data_plain)) / config['batch_size'])
            name_ordering = np.arange(len(train_data_plain), dtype=np.int32)
            np.random.shuffle(name_ordering)
            batch_start = 0

            # if it's to resume the previous training, reload the archive and settings before training
            if config['resume_training'] and epoch == 1:
                name_ordering, batch_id, loss, valid_param, optimizer_config = deserialize_from_file(config['training_archive'])
                batch_start += 1

                optimizer_config['rng'] = agent.rng
                optimizer_config['save'] = False
                optimizer_config['clipnorm'] = config['clipnorm']
                print('optimizer_config: %s' % str(optimizer_config))
                # agent.optimizer = optimizers.get(config['optimizer'], kwargs=optimizer_config)
                agent.optimizer.iterations.set_value(optimizer_config['iterations'])
                agent.optimizer.lr.set_value(optimizer_config['lr'])
                agent.optimizer.beta_1 = optimizer_config['beta_1']
                agent.optimizer.beta_2 = optimizer_config['beta_2']
                agent.optimizer.clipnorm = optimizer_config['clipnorm']
                # batch_start = 40001

            for batch_id in range(batch_start, num_batches):
                # 1. Prepare data
                data_ids = name_ordering[batch_id * config['batch_size']:min((batch_id + 1) * config['batch_size'], len(train_data_plain))]

                # obtain mini-batch data
                data_s = train_data_source[data_ids]
                data_t = train_data_target[data_ids]

                # convert one data (with multiple targets) into multiple ones
                data_s, data_t = split_into_multiple_and_padding(data_s, data_t)

                # 2. Training
                '''
                As the length of input varies often, it leads to frequent Out-of-Memory on GPU
                 Thus I have to segment each mini batch into mini-mini batches based on their lengths (number of words)
                 It slows down the speed somehow, but avoids the break-down effectively
                '''
                loss_batch = []

                mini_data_idx = 0
                max_size = config['mini_mini_batch_length'] # max length (#words) of each mini-mini batch
                stack_size = 0
                mini_data_s = []
                mini_data_t = []
                while mini_data_idx < len(data_s):
                    if len(data_s[mini_data_idx]) * len(data_t[mini_data_idx]) >= max_size:
                        logger.error('mini_mini_batch_length is too small. Enlarge it to 2 times')
                        max_size = len(data_s[mini_data_idx]) * len(data_t[mini_data_idx]) * 2
                        config['mini_mini_batch_length'] = max_size

                    # get a new mini-mini batch
                    while mini_data_idx < len(data_s) and stack_size + len(data_s[mini_data_idx]) * len(data_t[mini_data_idx]) < max_size:
                        mini_data_s.append(data_s[mini_data_idx])
                        mini_data_t.append(data_t[mini_data_idx])
                        stack_size += len(data_s[mini_data_idx]) * len(data_t[mini_data_idx])
                        mini_data_idx += 1
                    mini_data_s = np.asarray(mini_data_s)
                    mini_data_t = np.asarray(mini_data_t)

                    logger.info('Training minibatch %d/%d' % (mini_data_idx, len(data_s)))

                    # fit the mini-mini batch
                    if config['copynet']:
                        # mini_data_s=(batch_size, src_len), mini_data_t=(batch_size, trg_len), data_c=(batch_size, trg_len, src_len)
                        data_c = cc_martix(mini_data_s, mini_data_t)
                        loss_batch += [agent.train_(unk_filter(mini_data_s), unk_filter(mini_data_t), data_c)]
                        # loss += [agent.train_guard(unk_filter(mini_data_s), unk_filter(mini_data_t), data_c)]
                    else:
                        loss_batch += [agent.train_(unk_filter(mini_data_s), unk_filter(mini_data_t))]

                    mini_data_s = []
                    mini_data_t = []
                    stack_size  = 0

                # average the training loss and print progress
                mean_ll  = np.average(np.concatenate([l[0] for l in loss_batch]))
                mean_ppl = np.average(np.concatenate([l[1] for l in loss_batch]))
                loss.append([mean_ll, mean_ppl])
                progbar.update(batch_id, [('loss_reg', mean_ll),
                                          ('ppl.', mean_ppl)])

                # 3. Quick testing
                if config['do_quick_testing'] and batch_id % 200 == 0 and batch_id > 1:
                    print_case = '-' * 100 +'\n'

                    logger.info('Echo={} Evaluation Sampling.'.format(batch_id))
                    print_case += 'Echo={} Evaluation Sampling.\n'.format(batch_id)

                    logger.info('generating [training set] samples')
                    print_case += 'generating [training set] samples\n'

                    for _ in range(1):
                        idx              = int(np.floor(n_rng.rand() * train_size))

                        test_s_o, test_t_o = train_data_plain[idx]

                        if not config['multi_output']:
                            # create <abs, phrase> pair for each phrase
                            test_s, test_t = split_into_multiple_and_padding([test_s_o], [test_t_o])

                        inputs_unk = np.asarray(unk_filter(np.asarray(test_s[0], dtype='int32')), dtype='int32')
                        prediction, score = agent.generate_multiple(inputs_unk[None, :])

                        outs, metrics = agent.evaluate_multiple([test_s[0]], [test_t],
                                                                [test_s_o], [test_t_o],
                                                                [prediction], [score],
                                                                idx2word)
                        print('*' * 50)
                        print('*' * 50)

                    logger.info('generating [testing set] samples')
                    for _ in range(1):
                        idx            = int(np.floor(n_rng.rand() * test_size))
                        test_s_o, test_t_o = test_data_plain[idx]
                        if not config['multi_output']:
                            test_s, test_t = split_into_multiple_and_padding([test_s_o], [test_t_o])

                        inputs_unk = np.asarray(unk_filter(np.asarray(test_s[0], dtype='int32')), dtype='int32')
                        prediction, score = agent.generate_multiple(inputs_unk[None, :], return_all=False)

                        outs, metrics = agent.evaluate_multiple([test_s[0]], [test_t],
                                                                [test_s_o], [test_t_o],
                                                                [prediction], [score],
                                                                idx2word)
                        print('*' * 50)
                    # write examples to log file
                    with open(config['casestudy_log'], 'w+') as print_case_file:
                        print_case_file.write(print_case)

                # 4. Test on validation data for a few batches, and do early-stopping if needed
                if do_validate and batch_id % 1000 == 0 and not (batch_id==0 and epoch==1):
                    logger.info('Validate @ epoch=%d, batch=%d' % (epoch, batch_id))
                    # 1. Prepare data
                    data_s = np.array(validation_set['source'])
                    data_t = np.array(validation_set['target'])

                    # if len(data_s) > 2000:
                    #     data_s = data_s[:2000]
                    #     data_t = data_t[:2000]
                    # if not multi_output, split one data (with multiple targets) into multiple ones
                    if not config['multi_output']:
                        data_s, data_t = split_into_multiple_and_padding(data_s, data_t)

                    loss_valid = []

                    # for minibatch_id in range(int(math.ceil(len(data_s)/config['mini_batch_size']))):
                    #     mini_data_s = data_s[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_s))]
                    #     mini_data_t = data_t[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_t))]

                    mini_data_idx = 0
                    max_size = config['mini_mini_batch_length']
                    stack_size = 0
                    mini_data_s = []
                    mini_data_t = []
                    while mini_data_idx < len(data_s):
                        if len(data_s[mini_data_idx]) * len(data_t[mini_data_idx]) >= max_size:
                            logger.error('mini_mini_batch_length is too small. Enlarge it to 2 times')
                            max_size = len(data_s[mini_data_idx]) * len(data_t[mini_data_idx]) * 2
                            config['mini_mini_batch_length'] = max_size

                        while mini_data_idx < len(data_s) and stack_size + len(data_s[mini_data_idx]) * len(data_t[mini_data_idx]) < max_size:
                            mini_data_s.append(data_s[mini_data_idx])
                            mini_data_t.append(data_t[mini_data_idx])
                            stack_size += len(data_s[mini_data_idx]) * len(data_t[mini_data_idx])
                            mini_data_idx += 1
                        mini_data_s = np.asarray(mini_data_s)
                        mini_data_t = np.asarray(mini_data_t)

                        if config['copynet']:
                            data_c = cc_martix(mini_data_s, mini_data_t)
                            loss_valid += [agent.validate_(unk_filter(mini_data_s), unk_filter(mini_data_t), data_c)]
                        else:
                            loss_valid += [agent.validate_(unk_filter(mini_data_s), unk_filter(mini_data_t))]

                        if mini_data_idx % 100 == 0:
                            print('\t %d / %d' % (mini_data_idx, math.ceil(len(data_s))))

                        mini_data_s = []
                        mini_data_t = []
                        stack_size = 0

                    mean_ll = np.average(np.concatenate([l[0] for l in loss_valid]))
                    mean_ppl = np.average(np.concatenate([l[1] for l in loss_valid]))
                    logger.info('\tPrevious best score: \t ll=%f, \t ppl=%f' % (valid_param['valid_best_score'][0], valid_param['valid_best_score'][1]))
                    logger.info('\tCurrent score: \t ll=%f, \t ppl=%f' % (mean_ll, mean_ppl))

                    if mean_ll < valid_param['valid_best_score'][0]:
                        valid_param['valid_best_score'] = (mean_ll, mean_ppl)
                        logger.info('New best score')
                        valid_param['valids_not_improved'] = 0
                    else:
                        valid_param['valids_not_improved'] += 1
                        logger.info('Not improved for %s tests.' % valid_param['valids_not_improved'])

                # 5. Save model
                if batch_id % 500 == 0 and batch_id > 1:
                    # save the weights every K rounds
                    agent.save(config['path_experiment'] + '/experiments.{0}.id={1}.epoch={2}.batch={3}.pkl'.format(config['task_name'], config['timemark'], epoch, batch_id))

                    # save the game(training progress) in case of interrupt!
                    optimizer_config = agent.optimizer.get_config()
                    serialize_to_file([name_ordering, batch_id, loss, valid_param, optimizer_config], config['path_experiment'] + '/save_training_status.id={0}.epoch={1}.batch={2}.pkl'.format(config['timemark'], epoch, batch_id))
                    print(optimizer_config)
                    # agent.save_weight_json(config['path_experiment'] + '/weight.print.id={0}.epoch={1}.batch={2}.json'.format(config['timemark'], epoch, batch_id))

                # 6. Stop if exceed patience
                if valid_param['valids_not_improved']  >= valid_param['patience']:
                    print("Not improved for %s epochs. Stopping..." % valid_param['valids_not_improved'])
                    valid_param['early_stop'] = True
                    break

    '''
    test accuracy and f-score at the end of each epoch
    '''
    if do_predict:
        for dataset_name in config['testing_datasets']:
            # override the original test_set
            # if the dataset does not provide postag, use load_testing_data()
            test_set = keyphrase_test_dataset.testing_data_loader(dataset_name, kwargs=dict(basedir=config['path'])).load_testing_data(word2idx)
            # test_set = keyphrase_test_dataset.testing_data_loader(dataset_name, kwargs=dict(basedir=config['path'])).load_testing_data_postag(word2idx)
            # test_set = test_sets[dataset_name]

            test_data_plain = list(zip(*(test_set['source_str'], test_set['target_str'], test_set['source'], test_set['target'])))
            test_size = len(test_data_plain)

            print(dataset_name)
            print('Size of test data=%d' % test_size)
            print('Avg length=%d, Max length=%d' % (np.average([len(s) for s in test_set['source']]), np.max([len(s) for s in test_set['source']])))

            # use the first 400 samples in krapivin for testing
            if dataset_name == 'krapivin':
                test_data_plain = test_data_plain[:400]
                test_size = len(test_data_plain)

            progbar_test = Progbar(test_size, logger)
            logger.info('Predicting on %s' % dataset_name)

            input_encodings = []
            output_encodings = []

            predictions = []
            scores = []
            test_s_list = []
            test_t_list = []
            test_s_o_list = []
            test_t_o_list = []

            start_idx = -1

            # start_idx = 10800
            # test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, input_encodings, predictions, scores, output_encodings, idx2word = deserialize_from_file(config['predict_path'] + 'predict.{0}.{1}.id={2}.pkl'.format(config['predict_type'], dataset_name, start_idx))

            # Predict on testing data
            for idx in range(start_idx + 1, len(test_data_plain)): # len(test_data_plain)
                source_str, target_str, test_s_o, test_t_o = test_data_plain[idx]
                print('*'*20 + '  ' + str(idx)+ '  ' + '*'*20)
                # print(source_str)
                # print('[%d]%s' % (len(test_s_o), str(test_s_o)))
                # print(target_str)
                # print(test_t_o)
                # print('')

                if not config['multi_output']:
                    test_s, test_t = split_into_multiple_and_padding([test_s_o], [test_t_o])
                test_s = test_s[0]

                test_s_list.append(test_s)
                test_t_list.append(test_t)
                test_s_o_list.append(test_s_o)
                test_t_o_list.append(test_t_o)

                print('test_s_o=%d, test_t_o=%d, test_s=%d, test_t=%d' % (len(test_s_o), len(test_t_o), len(test_s), len(test_t)))

                inputs_unk = np.asarray(unk_filter(np.asarray(test_s, dtype='int32')), dtype='int32')
                # inputs_ = np.asarray(test_s, dtype='int32')


                if config['return_encoding']:
                    input_encoding, prediction, score, output_encoding = agent.generate_multiple(inputs_unk[None, :], return_all=True, return_encoding=True)
                    input_encodings.append(input_encoding)
                    output_encodings.append(output_encoding)
                else:
                    prediction, score = agent.generate_multiple(inputs_unk[None, :], return_encoding=False)

                # print(prediction)
                for p in prediction:
                    if any([f>=50000 for f in p]):
                        print(p)

                predictions.append(prediction)
                scores.append(score)
                progbar_test.update(idx, [])

                # temporary save
                if idx % 100 == 0:
                    print('Saving dump to: '+ config['predict_path'] + 'predict.{0}.{1}.id={2}.pkl'.format(config['predict_type'], dataset_name, idx))
                    serialize_to_file(
                        [test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, input_encodings, predictions,
                         scores, output_encodings, idx2word],
                        config['predict_path'] + 'predict.{0}.{1}.id={2}.pkl'.format(config['predict_type'], dataset_name, idx))

            # store predictions in file
            serialize_to_file([test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, input_encodings, predictions, scores, output_encodings, idx2word], config['predict_path'] + 'predict.{0}.{1}.pkl'.format(config['predict_type'], dataset_name))

    '''
    Evaluate on Testing Data
    '''
    if do_evaluate:
        for dataset_name in config['testing_datasets']:
            print_test = open(config['predict_path'] + '/experiments.{0}.id={1}.testing@{2}.{3}.len={4}.beam={5}.log'.format(config['task_name'],config['timemark'],dataset_name, config['predict_type'], config['max_len'], config['sample_beam']), 'w')

            test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, _, predictions, scores, _, idx2word = deserialize_from_file(config['predict_path']+'predict.{0}.{1}.pkl'.format(config['predict_type'], dataset_name))

            # use the first 400 samples in krapivin for testing
            if dataset_name == 'krapivin':
                new_test_set = {}
                for k,v in test_set.items():
                    new_test_set[k]  = v[:400]
                test_s_list     = test_s_list[:400] # numericalized source with padding
                test_t_list     = test_t_list[:400] # numericalized targets with padding
                test_s_o_list   = test_s_o_list[:400] # numericalized source without padding
                test_t_o_list   = test_t_o_list[:400] # numericalized targets without padding
                predictions     = predictions[:400]
                scores          = scores[:400]

                test_set = new_test_set

            print_test.write('Evaluating on %s size=%d @ epoch=%d \n' % (dataset_name, test_size, epoch))
            logger.info('Evaluating on %s size=%d @ epoch=%d \n' % (dataset_name, test_size, epoch))

            do_stem = True
            if dataset_name == 'semeval':
                do_stem = False

            # Evaluation
            outs, overall_score = keyphrase_utils.evaluate_multiple(config, test_set, test_s_list, test_t_list,
                                                        test_s_o_list, test_t_o_list,
                                                        predictions, scores, idx2word, do_stem,
                                                        model_name=config['task_name'], dataset_name=dataset_name)

            print_test.write(' '.join(outs))
            print_test.write(' '.join(['%s : %s' % (str(k), str(v)) for k,v in overall_score.items()]))
            logger.info('*' * 50)

            logger.info(overall_score)
            print_test.close()
