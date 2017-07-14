#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import re
import json

import nltk

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

SENTENCEDELIMITER = '.' #'<EOS>'
DIGIT = '<DIGIT>'

def export_Inspec_tokenized(dir_name, output_name):
    """
    Paper abstracts Inspec (Hulth, 2003)∗ #(Documents)=2,000,  #(Tokens/doc) <200, #(Keys/doc)=10
    Load data in seperate files and export to a json
    Only .uncontr is used (used in Hulth, 2003)
    """
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    total_length = 0
    total_keyphrases = 0
    total_occurance_keyphrases = 0

    output_list = []

    files = os.listdir(dir_name)
    count = 0

    for c, f in enumerate(files):
        if f.endswith('.abstr'):
            count += 1
            file_no = f[:f.index('.abstr')]
            text_file_name = f
            print("No. %s : file=%s, name=%s" % (c, f, file_no))

            keyphrase_file_name = file_no+'.uncontr'
            with open(dir_name+os.sep+text_file_name, 'r') as text_file:
                text = text_file.read().lower()
                text = text.replace('\n', SENTENCEDELIMITER, 1) # first line is title.
                text = re.sub('[\t\r\n]', ' ', text)
                text = (' '+SENTENCEDELIMITER+' ').join(sent_detector.tokenize(text))
                text = re.sub('\d+', '  ', text)
                text = filter(lambda w: len(w)>0, re.split('\W', text))
            with open(dir_name+os.sep+keyphrase_file_name, 'r') as keyphrase_file:
                keyphrases = keyphrase_file.read().lower()
                keyphrases = re.sub('[\t\r\n]', ' ', keyphrases)
                keyphrases = re.sub('\d+', ' '+DIGIT+' ', keyphrases)
                keyphrases = [re.sub('\W+', ' ', w.strip()) for w  in keyphrases.split(';')]

            for k in keyphrases:
                dict = {}
                dict['filename']=f
                dict['name']=k.split()
                dict['tokens']=text
                output_list.append(dict)

            clean_text = ' '.join(text)

            print(keyphrases)
            print(text)
            print(clean_text)
            print('text length = %s' % len(text))

            number_appearence = len(filter(lambda x: clean_text.find(x) > 0, keyphrases))
            print('keyphrase occurance = %s/%s' % (number_appearence,len(keyphrases)))
            print('--------------------------------------')

            total_length += len(text)
            total_keyphrases += len(keyphrases)
            total_occurance_keyphrases += number_appearence

    print('total documents = %s') % count
    print('average doc length = %s' % (float(total_length)/count))
    print('keyphrase occurance = %s/%s' % (total_occurance_keyphrases, total_keyphrases))
    print('averge keyphrase occurance = %s/%s' % (float(total_occurance_keyphrases)/count, float(total_keyphrases)/count))

    with open(output_name, 'w') as json_file:
        json_file.write(json.dumps(output_list))


def export_Inspec(Inspec_input_path, Inspec_output_path):
    '''
    export to json, without any preprocess
    :param Inspec_folder_name:
    :return:
    '''

    count = 0
    output_list = []
    for p, folders, docs in os.walk(Inspec_input_path):
        for f in docs:
            if f.endswith('.abstr'):
                count += 1
                file_no = f[:f.index('.abstr')]
                text_file_name = f
                print("No. %s : file=%s, name=%s" % (count, f, file_no))

                with open(os.path.join(p, f), 'r') as text_file:
                    text = text_file.read()
                    title = text[:text.find('\r')] # first line is title.
                    title = re.sub('[\t\r\n]', ' ', title).strip()
                    text = text[text.find('\r'):]
                    text = re.sub('[\t\r\n]', ' ', text).strip()

                keyphrase_file_name = file_no + '.uncontr'
                # keyphrase_file_name = file_no + '.contr'
                with open(os.path.join(p, keyphrase_file_name), 'r') as keyphrase_file:
                    keyphrases = keyphrase_file.read().lower()
                    keyphrases = re.sub('[\t\r\n]', ' ', keyphrases)
                    keyphrases = [re.sub('\W+', ' ', w.strip()) for w in keyphrases.split(';')]
                    keyphrases = ';'.join(keyphrases)

                dict = {}
                dict['title'] = title
                dict['keyword'] = keyphrases
                dict['abstract'] = text
                output_list.append(dict)

                print('\ttitle: %s' % dict['title'])
                print('\tabstract: %s' % text)
                print('\tkeyword: %s' % keyphrases)
                print('text length = %s' % len(text))
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

    with open(Inspec_output_path, 'w') as json_file:
        json_file.write(json.dumps(output_list))


BASE_DIR = os.path.realpath(os.path.curdir)+'/dataset/keyphrase/inspec/'

if __name__ == '__main__':
    os.chdir(BASE_DIR)
    # Inspec_folder_name = 'test'
    Inspec_input_name = 'all' # folder
    Inspec_output_name = 'inspec_all_tokenized.json'
    # export_Inspec(BASE_DIR+Inspec_input_name, BASE_DIR+Inspec_output_name)
    export_Inspec_tokenized(BASE_DIR+Inspec_input_name, BASE_DIR+Inspec_output_name)