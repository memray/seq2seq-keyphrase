#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python File Template 
"""
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize

import os
import re
import shutil

import nltk
import xml.etree.ElementTree as ET

import keyphrase.dataset.dataset_utils as utils
from emolga.utils.generic_utils import get_from_module

from keyphrase.config import setup_keyphrase_all
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
import numpy as np

from keyphrase.dataset import dataset_utils

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.text       = ''
        self.phrases    = []

    def __str__(self):
        return '%s\n\t%s\n\t%s' % (self.title, self.text, str(self.phrases))


class DataLoader(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.name    = self.__class__.__name__
        self.basedir = self.basedir
        self.doclist = []

    def get_docs(self, return_dict=True):
        '''
        :return: a list of dict instead of the Document object
        '''
        class_name = self.__class__.__name__.lower()
        if class_name == 'kdd' or class_name == 'www' or class_name == 'umd':
            self.load_xml(self.textdir)
        else:
            self.load_text(self.textdir)
            self.load_keyphrase(self.keyphrasedir)

        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name']        = d.name
            newd['abstract']    = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title']       = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword']     = ';'.join(d.phrases)
            doclist.append(newd)

        if return_dict:
            return doclist
        else:
            return self.doclist

    def __call__(self, idx2word, word2idx, type = 1):
        self.get_docs()

        pairs = []

        for doc in self.doclist:
            try:
                title= utils.get_tokens(doc.title, type)
                text = utils.get_tokens(doc.text, type)
                if type == 0:
                    title.append('<eos>')
                elif type == 1:
                    title.append('.')

                title.extend(text)
                text = title

                # trunk, many texts are too long, would lead to out-of-memory
                if len(text) > 1500:
                    text = text[:1500]

                keyphrases = [utils.get_tokens(k, type) for k in doc.phrases]
                pairs.append((text, keyphrases))

            except UnicodeDecodeError:
                print('UnicodeDecodeError detected! %s' % doc.name)
            # print(text)
            # print(keyphrases)
            # print('*'*50)
        dataset = utils.build_data(pairs, idx2word, word2idx)

        return dataset, self.doclist

    def load_xml(self, xmldir):
        '''
        for KDD/WWW/UMD only
        :return: doclist
        '''
        for filename in os.listdir(xmldir):
            with open(xmldir+filename) as textfile:
                doc = Document()
                doc.name = filename[:filename.find('.xml')]

                import string
                printable = set(string.printable)

                # print((filename))
                try:
                    lines = textfile.readlines()
                    xml = ''.join([filter(lambda x: x in printable, l) for l in lines])
                    root = ET.fromstring(xml)

                    doc.title = root.findall("title")[0].text
                    doc.abstract = root.findall("abstract")[0].text
                    doc.phrases = [n.text for n in root.findall("*/tag")]

                    self.doclist.append(doc)

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % filename )

    def load_text(self, textdir):
        for fid, filename in enumerate(os.listdir(textdir)):
            with open(textdir+filename) as textfile:
                doc = Document()
                doc.name = filename[:filename.find('.txt')]

                import string
                printable = set(string.printable)

                # print((textdir+filename))
                try:
                    lines = textfile.readlines()

                    lines = [list(filter(lambda x: x in printable, l)) for l in lines]

                    title = ''.join(lines[0]).encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # the 2nd line is abstract title
                    text  = (' '.join([''.join(l).strip() for l in lines[1:]])).encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # if lines[1].strip().lower() != 'abstract':
                    #     print('Wrong title detected : %s' % (filename))

                    doc.title = title
                    doc.text  = text
                    self.doclist.append(doc)

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % filename )

    def load_keyphrase(self, keyphrasedir):
        for did,doc in enumerate(self.doclist):
            phrase_set = set()
            if os.path.exists(self.keyphrasedir + doc.name + '.keyphrases'):
                with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
            # else:
            #     print(self.keyphrasedir + doc.name + '.keyphrases Not Found')

            if os.path.exists(self.keyphrasedir + doc.name + '.keywords'):
                with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
            # else:
            #     print(self.keyphrasedir + doc.name + '.keywords Not Found')

            doc.phrases = list(phrase_set)

    def load_testing_data_postag(self, word2idx):
        print('Loading testing dataset %s from %s' % (self.name, self.postag_datadir))
        text_file_paths = [self.text_postag_dir + n_ for n_ in os.listdir(self.text_postag_dir)]
        keyphrase_file_paths = [self.keyphrase_postag_dir + n_ for n_ in os.listdir(self.keyphrase_postag_dir)]

        def load_postag_text_(path):
            with open(path, 'r') as f:
                tokens = ' '.join(f.readlines()).split(' ')
                text = [t.split('_')[0] for t in tokens]
                postag = [t.split('_')[1] for t in tokens]
            return text, postag

        def load_keyphrase_(path):
            with open(path, 'r') as f:
                keyphrase_str = ';'.join([l.strip() for l in f.readlines()])
                return dataset_utils.process_keyphrase(keyphrase_str)

        texts = [load_postag_text_(f_) for f_ in text_file_paths]
        keyphrases = [load_keyphrase_(f_) for f_ in keyphrase_file_paths]

        instance = dict(source_str=[], target_str=[], source=[], source_postag=[], target=[], target_c=[])

        for (source, postag),target in zip(texts, keyphrases):
            A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]

            instance['source_str'] += [source]
            instance['target_str'] += [target]
            instance['source'] += [A]
            instance['source_postag'] += [postag]
            instance['target'] += [B]

        return instance

    def load_testing_data(self, word2idx):
        print('Loading testing dataset %s from %s' % (self.name, self.datadir))
        text_file_paths = [self.textdir + n_ for n_ in os.listdir(self.textdir) if n_.endswith('.txt')]
        # here is problematic. keep phrases in either '.txt' or 'keyphrases', but don't keep both files (kp20k uses both, delete either one would be fine).
        keyphrase_file_paths = [self.keyphrasedir + n_ for n_ in os.listdir(self.keyphrasedir) if n_.endswith('.keyphrases')]
        # keyphrase_file_paths = [self.keyphrasedir + n_ for n_ in os.listdir(self.keyphrasedir) if n_.endswith('.txt')]

        def _load_text(path):
            with open(path, 'r') as f:
                text = ' '.join(f.readlines()).split(' ')
            return text

        def _load_keyphrase(path):
            with open(path, 'r') as f:
                keyphrase_str = [l.strip().split(' ') for l in f.readlines()]
                return keyphrase_str

        texts = [_load_text(f_) for f_ in text_file_paths]
        keyphrases = [_load_keyphrase(f_) for f_ in keyphrase_file_paths]

        instance = dict(source_str=[], target_str=[], source=[], source_postag=[], target=[], target_c=[])

        for source, target in zip(texts, keyphrases):
            A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
            B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]

            instance['source_str'] += [source]
            instance['target_str'] += [target]
            instance['source'] += [A]
            instance['source_postag'] += [] # set to be empty
            instance['target'] += [B]

        return instance

class INSPEC(DataLoader):
    def __init__(self, **kwargs):
        super(INSPEC, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/INSPEC'
        # self.textdir = self.datadir + '/all_texts/'
        # self.keyphrasedir = self.datadir + '/gold_standard_keyphrases_2/'

        self.textdir = self.datadir + '/test_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_test/'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/inspec/'
        self.text_postag_dir = self.postag_datadir + 'text/'
        self.keyphrase_postag_dir = self.postag_datadir + 'keyphrase/'

class NUS(DataLoader):
    def __init__(self, **kwargs):
        super(NUS, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/NUS'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/nus/'
        self.text_postag_dir = self.postag_datadir + 'text/'
        self.keyphrase_postag_dir = self.postag_datadir + 'keyphrase/'

    def export(self):
        '''
        parse the original dataset into two folders: text and gold_standard_keyphrases
        :return:
        '''
        originaldir = self.datadir+'/original'
        for paper_id in os.listdir(originaldir):
            if os.path.isfile(originaldir+'/'+paper_id):
                continue
            # copy text to all_texts/
            text_file = originaldir+'/'+paper_id+'/'+paper_id+'.txt'
            shutil.copy2(text_file, self.textdir+'/'+paper_id+'.txt')
            # load keyphrases
            keyphrases = set()
            keyphrase_files = [originaldir+'/'+paper_id+'/'+paper_id+'.kwd']
            reader_phrase_dir = originaldir+'/'+paper_id+'/KEY/'
            for key_file in os.listdir(reader_phrase_dir):
                keyphrase_files.append(reader_phrase_dir+key_file)
            for key_file in keyphrase_files:
                with open(key_file, 'r') as f:
                    keyphrases.update(set([l.strip() for l in f.readlines()]))
            # write into gold_standard_keyphrases/
            if os.path.exists(self.keyphrasedir + paper_id + '.keyphrases'):
                with open(self.keyphrasedir + paper_id + '.keyphrases', 'w') as f:
                    for key in list(keyphrases):
                        if key.find(' ') != -1:
                            f.write(key+'\n')
            # else:
            #     print(self.keyphrasedir + paper_id + '.keyphrases Not Found')

            if os.path.exists(self.keyphrasedir + paper_id + '.keywords'):
                with open(self.keyphrasedir + paper_id + '.keywords', 'w') as f:
                    for key in list(keyphrases):
                        if key.find(' ') == -1:
                            f.write(key+'\n')
            # else:
            #     print(self.keyphrasedir + paper_id + '.keywords Not Found')
    def get_docs(self, only_abstract=True, return_dict=True):
        '''
        :return: a list of dict instead of the Document object
        The keyphrase in SemEval is already stemmed
        '''
        for filename in os.listdir(self.keyphrasedir):
            if not filename.endswith('keyphrases'):
                continue
            doc = Document()
            doc.name = filename[:filename.find('.keyphrases')]
            phrase_set = set()
            if os.path.exists(self.keyphrasedir + doc.name + '.keyphrases'):
                with open(self.keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
            # else:
            #     print(self.keyphrasedir + doc.name + '.keyphrases Not Found')

            if os.path.exists(self.keyphrasedir + doc.name + '.keywords'):
                with open(self.keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            doc.phrases = list(phrase_set)
            self.doclist.append(doc)

        for d in self.doclist:
            with open(self.textdir + d.name + '.txt', 'r') as f:
                import string
                printable = set(string.printable)

                # print((filename))
                try:
                    lines = f.readlines()

                    lines = [filter(lambda x: x in printable, l) for l in lines]

                    # 1st line is title
                    title = lines[0].encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # find abstract
                    index_abstract = None
                    for id, line in enumerate(lines):
                        if line.lower().strip().endswith('abstract') or line.lower().strip().startswith('abstract'):
                            index_abstract = id
                            break
                    if index_abstract == None:
                        print('abstract not found: %s' % d.name)
                        index_abstract = 1

                    # find introduction
                    if only_abstract:
                        index_introduction = len(lines)
                        for id, line in enumerate(lines):
                            if line.lower().strip().endswith('introduction'):
                                index_introduction = id
                                break
                        if index_introduction == len(lines):
                            print('Introduction not found: %s' % d.name)

                    # 2nd line is abstract title
                    text  = (' '.join(lines[2 : index_introduction])).encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # if lines[1].strip().lower() != 'abstract':
                    #     print('Wrong title detected : %s' % (filename))

                    d.title = title
                    d.text  = text

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % self.self.textdir + d.name + '.txt.final' )


        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name']        = d.name
            newd['abstract']    = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title']       = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword']     = ';'.join(d.phrases)
            doclist.append(newd)

        if return_dict:
            return doclist
        else:
            return self.doclist

class SemEval(DataLoader):
    def __init__(self, **kwargs):
        super(SemEval, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/SemEval'
        # self.textdir = self.datadir + '/all_texts/'
        # self.keyphrasedir = self.datadir + '/gold_standard_keyphrases_3/'
        self.textdir = self.datadir + '/test/'
        self.keyphrasedir = self.datadir + '/test_answer/test.combined.stem.final'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/semeval/'
        self.text_postag_dir = self.postag_datadir + 'text/'
        self.keyphrase_postag_dir = self.postag_datadir + 'keyphrase/'

    """
    def get_docs(self, only_abstract=True, returnDict=True):
        '''
        :return: a list of dict instead of the Document object
        The keyphrase in SemEval is already stemmed
        '''

        if self.keyphrasedir.endswith('test.combined.stem.final'):
            with open(self.keyphrasedir, 'r') as kp:
                lines = kp.readlines()
                for line in lines:
                    d = Document()
                    d.name = line[:line.index(':')].strip()
                    d.phrases = line[line.index(':')+1:].split(',')
                    self.doclist.append(d)

        for d in self.doclist:
            with open(self.textdir + d.name + '.txt', 'r') as f:
                import string
                printable = set(string.printable)

                # print((filename))
                try:
                    lines = f.readlines()

                    lines = [filter(lambda x: x in printable, l) for l in lines]

                    # 1st line is title
                    title = lines[0].encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # find abstract
                    index_abstract = None
                    for id, line in enumerate(lines):
                        if line.lower().strip().endswith('abstract') or line.lower().strip().startswith('abstract'):
                            index_abstract = id
                            break
                    if index_abstract == None:
                        print('abstract not found: %s' % d.name)
                        index_abstract = 1

                    # find introduction
                    if only_abstract:
                        index_introduction = len(lines)
                        for id, line in enumerate(lines):
                            if line.lower().strip().endswith('introduction'):
                                index_introduction = id
                                break
                        if index_introduction == len(lines):
                            print('Introduction not found: %s' % d.name)

                    # 2nd line is abstract title
                    text  = (' '.join(lines[2 : index_introduction])).encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # if lines[1].strip().lower() != 'abstract':
                    #     print('Wrong title detected : %s' % (filename))

                    d.title = title
                    d.text  = text

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % self.self.textdir + d.name + '.txt.final' )


        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name']        = d.name
            newd['abstract']    = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title']       = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword']     = ';'.join(d.phrases)
            doclist.append(newd)

        if returnDict:
            return doclist
        else:
            return self.doclist
    """

class KRAPIVIN(DataLoader):
    def __init__(self, **kwargs):
        super(KRAPIVIN, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/KRAPIVIN'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/krapivin/'
        self.text_postag_dir = self.postag_datadir + 'text/'
        self.keyphrase_postag_dir = self.postag_datadir + 'keyphrase/'

    def load_text(self, textdir):
        for filename in os.listdir(textdir):
            with open(textdir+filename) as textfile:
                doc = Document()
                doc.name = filename[:filename.find('.txt')]

                import string
                printable = set(string.printable)

                # print((filename))
                try:
                    lines = textfile.readlines()

                    lines = [list(filter(lambda x: x in printable, l)) for l in lines]

                    title = ''.join(lines[1]).encode('ascii', 'ignore').decode('ascii', 'ignore')
                    # the 2nd line is abstract title
                    text  = ''.join(lines[3]).encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # if lines[1].strip().lower() != 'abstract':
                    #     print('Wrong title detected : %s' % (filename))

                    doc.title = title
                    doc.text  = text
                    self.doclist.append(doc)

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % filename )

class KDD(DataLoader):
    def __init__(self, **kwargs):
        super(KDD, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/KDD'
        self.xmldir = self.datadir + '/acmparsed/'
        self.textdir = self.datadir + '/acmparsed/'
        self.keyphrasedir = self.datadir + '/acmparsed/'

class WWW(DataLoader):
    def __init__(self, **kwargs):
        super(WWW, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/WWW'
        self.xmldir = self.datadir + '/acmparsed/'
        self.textdir = self.datadir + '/acmparsed/'
        self.keyphrasedir = self.datadir + '/acmparsed/'

class UMD(DataLoader):
    def __init__(self, **kwargs):
        super(UMD, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/UMD'
        self.xmldir = self.datadir + '/acmparsed/'
        self.textdir = self.datadir + '/contentsubset/'
        self.keyphrasedir = self.datadir + '/gold/'

class DUC2001(DataLoader):
    def __init__(self, **kwargs):
        super(DUC2001, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/UMD'
        self.xmldir = self.datadir + '/acmparsed/'
        self.textdir = self.datadir + '/contentsubset/'
        self.keyphrasedir = self.datadir + '/gold/'

class DUC(DataLoader):
    def __init__(self, **kwargs):
        super(DUC, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/DUC/'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/duc/'
        self.text_postag_dir = self.postag_datadir + 'text/'
        self.keyphrase_postag_dir = self.postag_datadir + 'keyphrase/'

    def export_text_phrase(self):
        all_phrase_file = self.basedir + '/dataset/keyphrase/testing-data/DUC/DUC2001LabeledKeyphrase.txt'

        duc_set = set()
        with open(all_phrase_file, 'r') as all_pf:
            for line in all_pf.readlines():
                line    = line.strip()
                duc_id  = line[:line.find('@')].strip()
                phrases = filter(lambda x:len(x.strip()) > 0, line[line.find('@')+1 : ].split(';'))
                # print(duc_id)
                # print(phrases)

                with open(self.keyphrasedir + duc_id + '.keyphrases', 'w') as pf:
                    pf.write('\n'.join(phrases))

                duc_set.add(duc_id)

        # delete the irrelevant files
        count = 0
        for text_file in os.listdir(self.datadir + '/original/'):
            if text_file in duc_set:
                if text_file.startswith('AP'):
                    print('*' * 50)
                    print(text_file)
                count += 1

                with open(self.datadir + '/original/' + text_file, 'r') as tf:
                    source = ' '.join([l.strip() for l in tf.readlines()])

                    m = re.search(r'<HEAD>(.*?)</HEAD>', source, flags=re.IGNORECASE)
                    if m  == None:
                        m = re.search(r'<HEADLINE>(.*?)</HEADLINE>', source, flags=re.IGNORECASE)
                    if m  == None:
                        m = re.search(r'<HL>(.*?)</HL>', source, flags=re.IGNORECASE)
                    if m  == None:
                        m = re.search(r'<H3>(.*?)</H3>', source, flags=re.IGNORECASE)

                    title = m.group(1)
                    title = re.sub('<.*?>', '', title).strip()
                    if text_file.startswith('FT') and title.find('/') > 0:
                        title = title[title.find('/')+1:].strip()

                    m = re.search(r'<TEXT>(.*?)</TEXT>', source, flags=re.IGNORECASE)
                    text = m.group(1)
                    text = re.sub('<.*?>', '', text).strip()

                    if text_file.startswith('AP'):
                        print(title)
                        print(text)

                    with open(self.textdir + text_file + '.txt', 'w') as target_tf:
                        target_tf.write(title + '\n' + text)

            # else:
            #     print('Delete!')
            #     os.remove(self.textdir+text_file)
        print(count)

class KP20k(DataLoader):
    def __init__(self, **kwargs):
        super(KP20k, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/baseline-data/kp20k/'
        self.textdir = self.datadir + '/plain_text/'
        self.keyphrasedir = self.datadir + '/keyphrase/'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/kp20k/'
        self.text_postag_dir = self.postag_datadir + 'text/'
        self.keyphrase_postag_dir = self.postag_datadir + 'keyphrase/'

    def get_docs(self, return_dict=True):
        '''
        :return: a list of dict instead of the Document object
        '''
        for fname in os.listdir(self.textdir):
            d = Document()
            d.name = fname
            with open(self.textdir + fname, 'r') as textfile:
                lines = textfile.readlines()
                d.title = lines[0].strip()
                d.text = ' '.join(lines[1:])
            with open(self.keyphrasedir + fname, 'r') as phrasefile:
                d.phrases = [l.strip() for l in phrasefile.readlines()]
            self.doclist.append(d)

        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name'] = d.name
            newd['abstract'] = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title'] = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword'] = ';'.join(d.phrases)
            doclist.append(newd)

        if return_dict:
            return doclist
        else:
            return self.doclist

class KP2k_NEW(DataLoader):
    '''
    18,716 docs after filtering (no keyword etc)
    '''
    def __init__(self, **kwargs):
        super(KP2k_NEW, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/new_kp2k_for_theano_model/'
        self.textdir = self.datadir + '/text/'
        self.keyphrasedir = self.datadir + '/keyphrase/'

    def get_docs(self, return_dict=True):
        '''
        :return: a list of dict instead of the Document object
        '''
        for fname in os.listdir(self.textdir):
            d = Document()
            d.name = fname
            with open(self.textdir + fname, 'r') as textfile:
                lines = textfile.readlines()
                d.title = lines[0].strip()
                d.text = ' '.join(lines[1:])
            with open(self.keyphrasedir + fname, 'r') as phrasefile:
                d.phrases = [l.strip() for l in phrasefile.readlines()]
            self.doclist.append(d)

        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name'] = d.name
            newd['abstract'] = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title'] = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword'] = ';'.join(d.phrases)
            doclist.append(newd)

        if return_dict:
            return doclist
        else:
            return self.doclist

class IRBooks(DataLoader):
    def __init__(self, **kwargs):
        super(IRBooks, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/IRbooks/'
        self.textdir = self.datadir + '/ir_textbook.txt'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/IRbooks/'
        self.text_postag_dir = self.postag_datadir + 'text/'

    def get_docs(self, return_dict=True):
        '''
        :return: a list of dict instead of the Document object
        '''

        with open(self.textdir, 'r') as textfile:
            for line in textfile.readlines():
                d = Document()
                text_id         = line[:line.index('\t')]
                text_content    = line[line.index('\t'):]
                d.name          = text_id.strip()
                d.title         = text_content[:text_content.index('  ')].strip()
                d.text          = text_content[text_content.index('  '):].strip()
                d.phrases       = []

                if len(d.text.split()) > 2000:
                    print('[length=%d]%s - %s' % (len(d.text.split()), d.name, d.title))
                self.doclist.append(d)

                # with open(self.keyphrasedir + fname, 'r') as phrasefile:
                #     d.phrases = [l.strip() for l in phrasefile.readlines()]

        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name'] = d.name
            newd['abstract'] = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title'] = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword'] = ';'.join(d.phrases)
            doclist.append(newd)

        if return_dict:
            return doclist
        else:
            return self.doclist

class Quora(DataLoader):
    def __init__(self, **kwargs):
        super(Quora, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/Quora/'
        self.textdir = self.datadir + '/'

        self.postag_datadir = self.basedir + '/dataset/keyphrase/baseline-data/Quora/'
        self.text_postag_dir = self.postag_datadir + 'text/'

    def get_docs(self, return_dict=True):
        '''
        :return: a list of dict instead of the Document object
        '''
        for textfile_name in os.listdir(self.textdir):
            with open(self.textdir+textfile_name, 'r') as textfile:
                d = Document()
                d.name = textfile_name[:textfile_name.find('.')].strip()
                d.title = ''
                d.text = ' '.join([l.strip() for l in textfile.readlines()])
                d.phrases = []
                self.doclist.append(d)

        doclist = []
        for d in self.doclist:
            newd = {}
            newd['name'] = d.name
            newd['abstract'] = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title'] = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword'] = ';'.join(d.phrases)
            doclist.append(newd)

        if return_dict:
            return doclist
        else:
            return self.doclist


# aliases
inspec = INSPEC
nus = NUS
semeval = SemEval
krapivin = KRAPIVIN
kdd = KDD
www = WWW
umd = UMD
kp20k = KP20k
kp2k_new = KP2k_NEW
duc = DUC
irbooks = IRBooks
quora = Quora # for Runhua's data

def testing_data_loader(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    data_loader = get_from_module(identifier, globals(), 'data_loader', instantiate=True,
                           kwargs=kwargs)
    return data_loader

def load_additional_testing_data(testing_names, idx2word, word2idx, config, postagging=True, process_type=1):
    test_sets           = {}

    # rule out the ones appear in testing data
    for dataset_name in testing_names:

        if os.path.exists(config['path'] + '/dataset/keyphrase/'+config['data_process_name']+dataset_name+'.testing.pkl'):
            test_set = deserialize_from_file(config['path'] + '/dataset/keyphrase/'+config['data_process_name']+dataset_name+'.testing.pkl')
            print('Loading testing dataset %s from %s' % (dataset_name, config['path'] + '/dataset/keyphrase/'+config['data_process_name']+dataset_name+'.testing.pkl'))
        else:
            print('Creating testing dataset %s: %s' % (dataset_name, config['path'] + '/dataset/keyphrase/' + config[
                'data_process_name'] + dataset_name + '.testing.pkl'))
            dataloader          = testing_data_loader(dataset_name, kwargs=dict(basedir=config['path']))
            records             = dataloader.get_docs()
            records, pairs, _   = utils.load_pairs(records, process_type=process_type, do_filter=False)
            test_set            = utils.build_data(pairs, idx2word, word2idx)

            test_set['record']  = records

            if postagging:
                tagged_sources = get_postag_with_record(records, pairs)
                test_set['tagged_source']   = [[t[1] for t in s] for s in tagged_sources]

                if hasattr(dataloader, 'text_postag_dir') and dataloader.__getattribute__('text_postag_dir') != None:
                    print('Exporting postagged data to %s' % (dataloader.text_postag_dir))
                    if not os.path.exists(dataloader.text_postag_dir):
                        os.makedirs(dataloader.text_postag_dir)
                    for r_, p_, s_ in zip(records, pairs, tagged_sources):
                        with open(dataloader.text_postag_dir+ '/' + r_['name'] + '.txt', 'w') as f:
                            output_str = ' '.join([w+'_'+t for w,t in s_])
                            f.write(output_str)
                else:
                    print('text_postag_dir not found, no export of postagged data')

            serialize_to_file(test_set, config['path'] + '/dataset/keyphrase/'+config['data_process_name']+dataset_name+'.testing.pkl')

        test_sets[dataset_name] = test_set

    return test_sets


from nltk.stem.porter import *
from keyphrase.dataset import dataset_utils
def check_data():
    config = setup_keyphrase_all()
    train_set, validation_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])

    for dataset_name in config['testing_datasets']:
        print('*' * 50)
        print(dataset_name)

        number_groundtruth = 0
        number_present_groundtruth = 0

        loader = testing_data_loader(dataset_name, kwargs=dict(basedir = config['path']))

        if dataset_name == 'nus':
            docs   = loader.get_docs(only_abstract = True, return_dict=False)
        else:
            docs   = loader.get_docs(return_dict=False)

        stemmer = PorterStemmer()

        for id,doc in enumerate(docs):

            text_tokens = dataset_utils.get_tokens(doc.title.strip()+' '+ doc.text.strip())
            # if len(text_tokens) > 1500:
            #     text_tokens = text_tokens[:1500]
            print('[%d] length= %d' % (id, len(doc.text)))

            stemmed_input = [stemmer.stem(t).strip().lower() for t in text_tokens]

            phrase_str = ';'.join([l.strip() for l in doc.phrases])
            phrases = dataset_utils.process_keyphrase(phrase_str)
            targets = [[stemmer.stem(w).strip().lower() for w in target] for target in phrases]

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

        print('number_groundtruth='+str(number_groundtruth))
        print('number_present_groundtruth='+str(number_present_groundtruth))

        '''
        test_set, doclist = loader(idx2word, word2idx, type=0)
        test_data_plain = zip(*(test_set['source'], test_set['target'], doclist))

        for idx in xrange(len(test_data_plain)):  # len(test_data_plain)
            test_s_o, test_t_o, doc = test_data_plain[idx]
            target = doc.phrases

            if len(doc.text) < 50000:
                print('%d - %d : %d   \n\tname=%s, \n\ttitle=%s, \n\ttext=%s, \n\tlen(keyphrase)=%d' % (idx, len(test_s_o), len(test_t_o), doc.name, doc.title, doc.text, len(''.join(target))))
                print(doc)

            if (len(target)!=0 and len(''.join(target))/len(target) < 3):
                print('!' * 100)
                print('Error found')
                print('%d - %d : %d   name=%s, title=%d, text=%d, len(keyphrase)=%d' % (idx, len(test_s_o), len(test_t_o), doc.name, len(doc.title), len(doc.text), len(''.join(target))))
        '''

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

def get_postag_with_record(records, pairs):
    path = os.path.dirname(__file__)
    path = path[:path.rfind(os.sep, 0, len(path)-10)+1] + 'stanford-postagger/'
    print(path)
    # jar = '/Users/memray/Project/stanford/stanford-postagger/stanford-postagger.jar'
    jar = path + '/stanford-postagger.jar'
    model = path + '/models/english-bidirectional-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar)
    # model = '/Users/memray/Project/stanford/stanford-postagger/models/english-left3words-distsim.tagger'
    # model = '/Users/memray/Project/stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'

    stanford_dir = jar.rpartition('/')[0]
    stanford_jars = find_jars_within_path(stanford_dir)
    pos_tagger._stanford_jar = ':'.join(stanford_jars)

    tagged_source = []
    # Predict on testing data
    for idx, (record, pair) in enumerate(zip(records, pairs)):  # len(test_data_plain)
        print('*' * 100)
        print('File: '  + record['name'])
        print('Input: ' + str(pair[0]))
        text = pos_tagger.tag(pair[0])
        print('[%d/%d][%d] : %s' % (idx, len(records) , len(pair[0]), str(text)))
        tagged_source.append(text)

    return tagged_source


def get_postag_with_index(sources, idx2word, word2idx):
    path = os.path.dirname(__file__)
    path = path[:path.rfind(os.sep, 0, len(path)-10)+1] + 'stanford-postagger/'
    print(path)
    # jar = '/Users/memray/Project/stanford/stanford-postagger/stanford-postagger.jar'
    jar = path + '/stanford-postagger.jar'
    model = path + '/models/english-bidirectional-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar)
    # model = '/Users/memray/Project/stanford/stanford-postagger/models/english-left3words-distsim.tagger'
    # model = '/Users/memray/Project/stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'

    stanford_dir = jar.rpartition('/')[0]
    stanford_jars = find_jars_within_path(stanford_dir)
    pos_tagger._stanford_jar = ':'.join(stanford_jars)

    tagged_source = []
    # Predict on testing data
    for idx in xrange(len(sources)):  # len(test_data_plain)
        test_s_o = sources[idx]
        source_text = keyphrase_utils.cut_zero(test_s_o, idx2word)
        text = pos_tagger.tag(source_text)
        print('[%d/%d] : %s' % (idx, len(sources), str(text)))

        tagged_source.append(text)

    return tagged_source

def check_postag(config):
    train_set, validation_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])

    path = os.path.dirname(__file__)
    path = path[:path.rfind(os.sep, 0, len(path)-10)+1] + 'stanford-postagger/'
    jar = path + '/stanford-postagger.jar'
    model = path + '/models/english-bidirectional-distsim.tagger'
    pos_tagger = StanfordPOSTagger(model, jar)

    for dataset_name in config['testing_datasets']:
        # override the original test_set
        # test_set = load_testing_data(dataset_name, kwargs=dict(basedir=config['path']))(idx2word, word2idx, config['preprocess_type'])

        test_sets = load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config)
        test_set = test_sets[dataset_name]

        # print(dataset_name)
        # print('Avg length=%d, Max length=%d' % (np.average([len(s) for s in test_set['source']]), np.max([len(s) for s in test_set['source']])))
        test_data_plain = zip(*(test_set['source'], test_set['target']))

        test_size = len(test_data_plain)

        # Alternatively to setting the CLASSPATH add the jar and model via their path:
        jar = '/Users/memray/Project/stanford/stanford-postagger/stanford-postagger.jar'
        # model = '/Users/memray/Project/stanford/stanford-postagger/models/english-left3words-distsim.tagger'
        model = '/Users/memray/Project/stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'
        pos_tagger = StanfordPOSTagger(model, jar)

        for idx in xrange(len(test_data_plain)):  # len(test_data_plain)
            test_s_o, test_t_o = test_data_plain[idx]

            source = keyphrase_utils.cut_zero(test_s_o, idx2word)

            print(source)

            # Add other jars from Stanford directory
            stanford_dir = jar.rpartition('/')[0]
            stanford_jars = find_jars_within_path(stanford_dir)
            pos_tagger._stanford_jar = ':'.join(stanford_jars)

            text = pos_tagger.tag(source)
            print(text)

if __name__ == '__main__':
    # config = setup_keyphrase_all()
    #
    # loader = testing_data_loader('duc', kwargs=dict(basedir=config['path']))
    # loader.export_text_phrase()
    # docs   = loader.get_docs()

    check_data()
    pass

    # check_postag(config)

    # train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    # load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config)

            # if len(test_t_o) < 3:
            #
            #     doc.text = re.sub(r'[\r\n\t]', ' ', doc.text)
            #     print('name:\t%s' % doc.name)
            #     print('text:\t%s' % doc.text)
            #     print('phrase:\t%s' % str(doc.phrases))
            # if idx % 100 == 0:
            #     print(test_data_plain[idx])