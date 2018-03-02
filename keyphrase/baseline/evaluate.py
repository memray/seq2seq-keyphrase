import math
import logging
import string

import scipy
from nltk.stem.porter import *
import numpy as np

import os
import sys
import keyphrase.config as config
# prepare logging.
from keyphrase.dataset import dataset_utils
import keyphrase.config

# config = keyphrase.config.setup_keyphrase_all()
config = keyphrase.config.setup_keyphrase_baseline()  # load settings.

def load_phrase(file_path, tokenize=True):
    phrases = []
    with open(file_path, 'r') as f:
        # TODO here the ground-truth is already after processing, contains <digit>, not good for baseline methods...
        if tokenize:
            phrase_str = ';'.join([l.strip() for l in f.readlines()])
            phrases = dataset_utils.process_keyphrase(phrase_str)
        else:
            phrases = [l.strip().split(' ') for l in f.readlines()]
        return phrases

def evaluate_(text_dir, target_dir, prediction_dir, model_name, dataset_name, do_stem=True):
    '''
    '''
    stemmer = PorterStemmer()

    print('Evaluating on %s@%s' % (model_name, dataset_name))
    # Evaluation part
    micro_metrics = []
    micro_matches = []

    doc_names = [name[:name.index('.')] for name in os.listdir(text_dir)]

    number_groundtruth = 0
    number_present_groundtruth = 0

    for doc_name in doc_names:
        logger.info('[FILE]{0}'.format(text_dir+'/'+doc_name+'.txt'))
        with open(text_dir+'/'+doc_name+'.txt', 'r') as f:
            text_tokens = (' '.join(f.readlines())).split( )

            text    = [t.split('_')[0] for t in text_tokens]
            postag  = [t.split('_')[1] for t in text_tokens]

        targets = load_phrase(target_dir+'/'+doc_name+'.txt', True)

        predictions = load_phrase(prediction_dir+'/'+doc_name+'.txt.phrases', False)

        # do processing to baseline predictions
        if (not model_name.startswith('CopyRNN')) and (not model_name.startswith('RNN')):
            predictions = dataset_utils.process_keyphrase(';'.join([' '.join(p) for p in predictions]))

        correctly_matched = np.asarray([0] * len(predictions), dtype='int32')

        print(targets)
        print(predictions)
        print('*' * 100)

        # convert target index into string
        if do_stem:
            stemmed_input    = [stemmer.stem(t).strip().lower() for t in text]
            targets = [[stemmer.stem(w).strip().lower() for w in target] for target in targets]

        if 'target_filter' in config:
            present_targets = []

            for target in targets:
                keep = True
                # whether do filtering on groundtruth phrases. if config['target_filter']==None, do nothing
                match = None
                for i in range(len(stemmed_input) - len(target) + 1):
                    match = None
                    for j in range(len(target)):
                        if target[j] != stemmed_input[i + j]:
                            match = False
                            break
                    if j == len(target) - 1 and match == None:
                        match = True
                        break

                if match == True:
                    # if match and 'appear-only', keep this phrase
                    if config['target_filter'] == 'appear-only':
                        keep = keep and True
                    elif config['target_filter'] == 'non-appear-only':
                        keep = keep and False
                elif match == False:
                    # if not match and 'appear-only', discard this phrase
                    if config['target_filter'] == 'appear-only':
                        keep = keep and False
                    # if not match and 'non-appear-only', keep this phrase
                    elif config['target_filter'] == 'non-appear-only':
                        keep = keep and True

                if not keep:
                    continue

                present_targets.append(target)

            number_groundtruth += len(targets)
            number_present_groundtruth += len(present_targets)
            targets = present_targets

        printable = set(string.printable)
        # lines = [filter(lambda x: x in printable, l) for l in lines]
        predictions = [[filter(lambda x: x in printable, w) for w in prediction] for prediction in predictions]
        predictions = [[stemmer.stem(w).strip().lower() for w in prediction] for prediction in predictions]
        for pid, predict in enumerate(predictions):
            # check whether the predicted phrase is correct (match any groundtruth)
            for target in targets:
                if len(target) == len(predict):
                    flag = True
                    for i, w in enumerate(predict):
                        if predict[i] != target[i]:
                            flag = False
                    if flag:
                        correctly_matched[pid] = 1
                        break

        metric_dict = {}
        for number_to_predict in [5, 10, 15]:
            metric_dict['target_number'] = len(targets)
            metric_dict['prediction_number'] = len(predictions)
            metric_dict['correct_number@%d' % number_to_predict] = sum(correctly_matched[:number_to_predict])

            metric_dict['p@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict])) / float(
                number_to_predict)

            if len(targets) != 0:
                metric_dict['r@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict])) / float(
                    len(targets))
            else:
                metric_dict['r@%d' % number_to_predict] = 0

            if metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict] != 0:
                metric_dict['f1@%d' % number_to_predict] = 2 * metric_dict['p@%d' % number_to_predict] * metric_dict[
                    'r@%d' % number_to_predict] / float(
                    metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict])
            else:
                metric_dict['f1@%d' % number_to_predict] = 0

            # Compute the binary preference measure (Bpref)
            bpref = 0.
            trunked_match = correctly_matched[:number_to_predict].tolist()  # get the first K prediction to evaluate
            match_indexes = np.nonzero(trunked_match)[0]

            if len(match_indexes) > 0:
                for mid, mindex in enumerate(match_indexes):
                    bpref += 1. - float(mindex - mid) / float(
                        number_to_predict)  # there're mindex elements, and mid elements are correct, before the (mindex+1)-th element
                metric_dict['bpref@%d' % number_to_predict] = float(bpref) / float(len(match_indexes))
            else:
                metric_dict['bpref@%d' % number_to_predict] = 0

            # Compute the mean reciprocal rank (MRR)
            rank_first = 0
            try:
                rank_first = trunked_match.index(1) + 1
            except ValueError:
                pass

            if rank_first > 0:
                metric_dict['mrr@%d' % number_to_predict] = float(1) / float(rank_first)
            else:
                metric_dict['mrr@%d' % number_to_predict] = 0

        micro_metrics.append(metric_dict)
        micro_matches.append(correctly_matched)

        '''
        Print information on each prediction
        '''
        # print stuff
        a = '[SOURCE][{0}]: {1}'.format(len(text) ,' '.join(text))
        logger.info(a)
        a += '\n'

        b = '[TARGET]: %d targets\n\t\t' % (len(targets))
        for id, target in enumerate(targets):
            b += ' '.join(target) + '; '
        logger.info(b)
        b += '\n'
        c = '[DECODE]: %d predictions' % (len(predictions))
        for id, predict in enumerate(predictions):
            c += ('\n\t\t[%d][%d]' % (len(predict), sum([len(w) for w in predict]))) + ' '.join(predict)
            if correctly_matched[id] == 1:
                c += ' [correct!]'
                # print(('\n\t\t[%.3f]'% score) + ' '.join(predict) + ' [correct!]')
                # print(('\n\t\t[%.3f]'% score) + ' '.join(predict))
        c += '\n'

        # c = '[DECODE]: {}'.format(' '.join(cut_zero(phrase, idx2word)))
        # if inputs_unk is not None:
        #     k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
        #     logger.info(k)
        # a += k
        logger.info(c)
        a += b + c

        for number_to_predict in [5, 10, 15]:
            d = '@%d - Precision=%.4f, Recall=%.4f, F1=%.4f, Bpref=%.4f, MRR=%.4f' % (
            number_to_predict, metric_dict['p@%d' % number_to_predict], metric_dict['r@%d' % number_to_predict],
            metric_dict['f1@%d' % number_to_predict], metric_dict['bpref@%d' % number_to_predict], metric_dict['mrr@%d' % number_to_predict])
            logger.info(d)
            a += d + '\n'

        logger.info('*' * 100)

    logger.info('#(Ground-truth Keyphrase)=%d' % number_groundtruth)
    logger.info('#(Present Ground-truth Keyphrase)=%d' % number_present_groundtruth)

    '''
    Export the f@5 and f@10 for significance test
    '''
    for k in [5, 10]:
        with open(config['predict_path'] + '/micro-f@%d-' % (k) + model_name+'-'+dataset_name+'.txt', 'w') as writer:
            writer.write('\n'.join([str(m['f1@%d' % k]) for m in micro_metrics]))

    '''
    Compute the corpus evaluation
    '''
    csv_writer = open(config['predict_path'] + '/evaluate-' + model_name+'-'+dataset_name+'.txt', 'w')

    real_test_size = len(doc_names)
    overall_score = {}
    for k in [5, 10, 15]:
        correct_number = sum([m['correct_number@%d' % k] for m in micro_metrics])
        overall_target_number = sum([m['target_number'] for m in micro_metrics])
        overall_prediction_number = sum([m['prediction_number'] for m in micro_metrics])

        if real_test_size * k < overall_prediction_number:
            overall_prediction_number = real_test_size * k

        # Compute the Micro Measures, by averaging the micro-score of each prediction
        overall_score['p@%d' % k] = float(sum([m['p@%d' % k] for m in micro_metrics])) / float(real_test_size)
        overall_score['r@%d' % k] = float(sum([m['r@%d' % k] for m in micro_metrics])) / float(real_test_size)
        overall_score['f1@%d' % k] = float(sum([m['f1@%d' % k] for m in micro_metrics])) / float(real_test_size)

        # Print basic statistics
        logger.info('%s@%s' % (model_name, dataset_name))
        output_str = 'Overall - %s valid testing data=%d, Number of Target=%d/%d, Number of Prediction=%d, Number of Correct=%d' % (
                    config['predict_type'], real_test_size,
                    overall_target_number, overall_target_number,
                    overall_prediction_number, correct_number
        )
        logger.info(output_str)
        # Print micro-average performance
        output_str = 'Micro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['p@%d' % k],
                    k, overall_score['r@%d' % k],
                    k, overall_score['f1@%d' % k]
        )
        logger.info(output_str)
        csv_writer.write(', %f, %f, %f' % (
                    overall_score['p@%d' % k],
                    overall_score['r@%d' % k],
                    overall_score['f1@%d' % k]
        ))

        # Print macro-average performance
        overall_score['macro_p@%d' % k] = correct_number / float(overall_prediction_number)
        overall_score['macro_r@%d' % k] = correct_number / float(overall_target_number)
        if overall_score['macro_p@%d' % k] + overall_score['macro_r@%d' % k] > 0:
            overall_score['macro_f1@%d' % k] = 2 * overall_score['macro_p@%d' % k] * overall_score[
                'macro_r@%d' % k] / float(overall_score['macro_p@%d' % k] + overall_score['macro_r@%d' % k])
        else:
            overall_score['macro_f1@%d' % k] = 0

        output_str = 'Macro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['macro_p@%d' % k],
                    k, overall_score['macro_r@%d' % k],
                    k, overall_score['macro_f1@%d' % k]
        )
        logger.info(output_str)
        csv_writer.write(', %f, %f, %f' % (
                    overall_score['macro_p@%d' % k],
                    overall_score['macro_r@%d' % k],
                    overall_score['macro_f1@%d' % k]
        ))

        # Compute the binary preference measure (Bpref)
        overall_score['bpref@%d' % k] = float(sum([m['bpref@%d' % k] for m in micro_metrics])) / float(real_test_size)

        # Compute the mean reciprocal rank (MRR)
        overall_score['mrr@%d' % k] = float(sum([m['mrr@%d' % k] for m in micro_metrics])) / float(real_test_size)

        output_str = '\t\t\tBpref@%d=%f, MRR@%d=%f' % (
                    k, overall_score['bpref@%d' % k],
                    k, overall_score['mrr@%d' % k]
        )
        logger.info(output_str)
    csv_writer.close()



def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
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

print('Log path: %s' % (
    config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'], config['timemark'])))
logger = init_logging(
    config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'], config['timemark']))
logger = logging.getLogger(__name__)




def evaluate_baselines():
    '''
    evaluate baselines' performance
    :return:
    '''
    # base_dir = '/Users/memray/Project/Keyphrase_Extractor-UTD/'
    # 'TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'Maui', 'KEA', 'RNN_present', 'CopyRNN_present_singleword=0', 'CopyRNN_present_singleword=1', 'CopyRNN_present_singleword=2'
    models = ['CopyRNN_present_singleword=1']

    test_sets = config['testing_datasets']

    for model_name in models:
        for dataset_name in test_sets:
            text_dir       = config['baseline_data_path'] + dataset_name + '/text/'
            target_dir     = config['baseline_data_path'] + dataset_name + '/keyphrase/'

            base_dir = config['path'] + '/dataset/keyphrase/prediction/' + model_name + '/'
            prediction_dir = base_dir + dataset_name

            #if model_name == 'Maui':
            #    prediction_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/maui_output/' + dataset_name
            #if model_name == 'Kea':
            #    prediction_dir = '/Users/memray/Project/seq2seq-keyphrase/dataset/keyphrase/baseline-data/maui/kea_output/' + dataset_name

            evaluate_(text_dir, target_dir, prediction_dir, model_name, dataset_name)

def significance_test():
    model1 = 'CopyRNN'
    models = ['TfIdf', 'TextRank', 'SingleRank', 'ExpandRank', 'RNN', 'CopyRNN']

    test_sets = config['testing_datasets']

    def load_result(filepath):
        with open(filepath, 'r') as reader:
            return [float(l.strip()) for l in reader.readlines()]

    for model2 in models:
        print('*'*20 + '  %s Vs. %s  ' % (model1, model2) + '*' * 20)
        for dataset_name in test_sets:
            for k in [5, 10]:
                print('Evaluating on %s@%d' % (dataset_name, k))
                filepath = config['predict_path'] + '/micro-f@%d-' % (k) + model1 + '-' + dataset_name + '.txt'
                val1 = load_result(filepath)
                filepath = config['predict_path'] + '/micro-f@%d-' % (k) + model2 + '-' + dataset_name + '.txt'
                val2 = load_result(filepath)
                s_test = scipy.stats.wilcoxon(val1, val2)
                print(s_test)

if __name__ == '__main__':
    evaluate_baselines()
    # significance_test()
