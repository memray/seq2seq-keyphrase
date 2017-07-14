import json

__author__ = 'jiataogu'
import numpy as np
import numpy.random as rng
import cPickle as pickle
import pprint
import sys
import hickle

from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes
from fuel import streams

def serialize_to_file_json(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    f = open(path, 'w')
    json.dump(obj, f)
    f.close()

def serialize_to_file_hdf5(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    f = open(path, 'w')
    hickle.dump(obj, f)
    f.close()

def serialize_to_file(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    print('serialize to %s' % path)
    f = open(path, 'wb')
    pickle.dump(obj, f, protocol=protocol)
    f.close()


def show_txt(array, path):
    f = open(path, 'w')
    for line in array:
        f.write(' '.join(line) + '\n')

    f.close()


def divide_dataset(dataset, test_size, max_size):
    train_set = dict()
    test_set  = dict()

    for w in dataset:
        train_set[w] = dataset[w][test_size:max_size].astype('int32')
        test_set[w]  = dataset[w][:test_size].astype('int32')

    return train_set, test_set

def deserialize_from_file_json(path):
    f = open(path, 'r')
    obj = json.load(f)
    f.close()
    return obj

def deserialize_from_file_hdf5(path):
    f = open(path, 'r')
    obj = hickle.load(f)
    f.close()
    return obj

def deserialize_from_file(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def build_fuel(data):
    # create fuel dataset.
    dataset     = datasets.IndexableDataset(indexables=OrderedDict([('data', data)]))
    dataset.example_iteration_scheme \
                = schemes.ShuffledExampleScheme(dataset.num_examples)
    return dataset, len(data)


def obtain_stream(dataset, batch_size, size=1):
    if size == 1:
        data_stream = dataset.get_example_stream()
        data_stream = transformers.Batch(data_stream, iteration_scheme=schemes.ConstantScheme(batch_size))

        # add padding and masks to the dataset
        data_stream = transformers.Padding(data_stream, mask_sources=('data'))
        return data_stream
    else:
        data_streams = [dataset.get_example_stream() for _ in range(size)]
        data_streams = [transformers.Batch(data_stream, iteration_scheme=schemes.ConstantScheme(batch_size))
                        for data_stream in data_streams]
        data_streams = [transformers.Padding(data_stream, mask_sources=('data')) for data_stream in data_streams]
        return data_streams

def build_ptb():
    path = './ptbcorpus/'
    print(path)
    # make the dataset and vocabulary
    X_train = [l.split() for l in open(path + 'ptb.train.txt').readlines()]
    X_test  = [l.split() for l in open(path + 'ptb.test.txt').readlines()]
    X_valid = [l.split() for l in open(path + 'ptb.valid.txt').readlines()]

    X = X_train + X_test + X_valid
    idx2word    = dict(enumerate(set([w for l in X for w in l]), 1))
    idx2word[0] = '<eol>'
    word2idx    = {v: k for k, v in idx2word.items()}
    ixwords_train = [[word2idx[w] for w in l] for l in X_train]
    ixwords_test  = [[word2idx[w] for w in l] for l in X_test]
    ixwords_valid = [[word2idx[w] for w in l] for l in X_valid]
    ixwords_tv    = [[word2idx[w] for w in l] for l in (X_train + X_valid)]

    max_len = max([len(w) for w in X_train])
    print(max_len)
    # serialization:
    # serialize_to_file(ixwords_train, path + 'data_train.pkl')
    # serialize_to_file(ixwords_test,  path + 'data_test.pkl')
    # serialize_to_file(ixwords_valid, path + 'data_valid.pkl')
    # serialize_to_file(ixwords_tv,    path + 'data_tv.pkl')
    # serialize_to_file([idx2word, word2idx], path + 'voc.pkl')
    # show_txt(X, 'data.txt')
    print('save done.')


def filter_unk(X, min_freq=5):
    voc = dict()
    for l in X:
        for w in l:
            if w not in voc:
                voc[w]  = 1
            else:
                voc[w] += 1

    word2idx   = dict()
    word2idx['<eol>'] = 0
    id2word    = dict()
    id2word[0] = '<eol>'

    at         = 1
    for w in voc:
        if voc[w] > min_freq:
            word2idx[w] = at
            id2word[at] = w
            at += 1

    word2idx['<unk>'] = at
    id2word[at] = '<unk>'
    return word2idx, id2word


def build_msr():
    # path = '/home/thoma/Work/Dial-DRL/dataset/MSRSCC/'
    path = '/Users/jiataogu/Work/Dial-DRL/dataset/MSRSCC/'
    print(path)

    X           = [l.split() for l in open(path + 'train.txt').readlines()]
    word2idx, idx2word = filter_unk(X, min_freq=5)
    print('vocabulary size={0}. {1} samples'.format(len(word2idx), len(X)))

    mean_len = np.mean([len(w) for w in X])
    print('mean len = {}'.format(mean_len))

    ixwords     = [[word2idx[w]
                    if w in word2idx
                    else word2idx['<unk>']
                    for w in l] for l in X]
    print(ixwords[0])
    # serialization:
    serialize_to_file(ixwords, path + 'data_train.pkl')


if __name__ == '__main__':
    build_msr()
    # build_ptb()
    # build_dataset()
    # game = GuessOrder(size=8)
    # q = 'Is there any number smaller de than 6 in the last 3 numbers ?'
    # print(game.easy_parse(q))

