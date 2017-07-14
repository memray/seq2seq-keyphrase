#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Load the paper metadata from json, do preprocess (cleanup, tokenization for words and sentences) and export to json
'''
import json
import re

import sys
import nltk

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def load_file(input_path):
    record_dict = {}
    count = 0
    no_keyword_abstract = 0
    with open(input_path, 'r') as f:
        for line in f:
            # clean the string
            line = re.sub(r'<.*?>|\s+|', ' ', line.lower())
            # load object
            record = json.loads(line)

            # store the name of input sentence
            text = record['abstract']
            text = re.sub('[\t\r\n]', ' ', text)
            text = record['title'] + ' <EOS> ' + ' <EOS> '.join(sent_detector.tokenize(text)) + ' <EOS>'
            text = filter(lambda w: len(w)>0, re.split('\W',text))
            record['tokens'] = text

            # store the terms of outputs
            keyphrases = record['keyword']
            record['name'] = [filter(lambda w: len(w)>0, re.split('\W',phrase)) for phrase in keyphrases.split(';')]

            # filter out the duplicate
            record_dict[record['title']] = record

            count += 1
            if len(record['keyword'])==0 or len(record['abstract'])==0:
                no_keyword_abstract += 0

            record['filename'] = record['title']
            print(record['title'])
            # print(record['abstract'])
            # print(record['token'])
            # print(record['keyword'])
            # print(record['name'])
            # print('')

    print('Total paper = %d' % count)
    print('Remove duplicate = %d' % len(record_dict))
    print('No abstract/keyword = %d' % no_keyword_abstract)
    return record_dict.values()

'''
Two ways to preprocess, -d 0 will export one abstract to one phrase, -d 1 will export one abstract to multiple phrases
'''
if __name__ == '__main__':
    sys.argv = 'dataset/keyphrase/million-paper/all_title_abstract_keyword.json dataset/keyphrase/million-paper/processed_all_title_abstract_keyword_one2many.json 1'.split()

    if len(sys.argv) < 3:
        print 'Usage <keyword_input_json_file> <output_file> -d 0|1 \n' \
              '     -d: format to output, 0 means one abstract to one keyphrase, 1 means one to many keyphrases'
        sys.exit(-1)

    input_file = sys.argv[0]
    output_file = sys.argv[1]
    records = load_file(input_file)

    print(len(records))

    with open(output_file, 'w') as out:
        output_list = []
        if sys.argv[2]=='0':
            for record in records:
                for keyphrase in record['name']:
                    dict = {}
                    dict['tokens'] = record['tokens']
                    dict['name'] = keyphrase
                    dict['filename'] = record['filename']
                    output_list.append(dict)
        if sys.argv[2]=='1':
            for record in records:
                dict = {}
                dict['tokens'] = record['tokens']
                dict['name'] = record['name']
                dict['filename'] = record['filename']
                output_list.append(dict)
        print(len(output_list))
        out.write(json.dumps(output_list))


