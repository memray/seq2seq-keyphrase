#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check how many non-duplicate and valid (some doesn't contain title/keyword/abstract) items in the data
"""
import json
import re
import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    PATH = os.path.realpath(os.path.curdir)+'/dataset/keyphrase/million-paper/'
    data_name = 'ALL'

    data_path = {'ALL':PATH+'raw_data/'+'all_title_abstract_keyword.json',
                 'ACM':PATH+'raw_data/'+'acm_title_abstract_keyword.json'}
    export_path = {'ALL':PATH+'all_title_abstract_keyword_clean.json',
                   'ACM':PATH+'acm_title_abstract_keyword_clean.json'}

    wordlist_path = data_path[data_name]

    # load data and filter invalid items
    title_map = {}
    count = 0
    no_keyword_abstract = 0
    with open(wordlist_path, 'r') as f:
        for line in f:
            line = re.sub(r'<.*?>|\s+|', ' ', line)

            d = json.loads(line)

            count += 1

            # line['title'] = line['title'].replace('',' ').replace('    ',' ')
            # print(d['title'])
            # print(d['keyword'].split(';'))
            # print(d['abstract'])

            if 'title' not in d or 'keyword' not in d or 'abstract' not in d :
                no_keyword_abstract += 1
                continue
            title_map[d['title']] = d

    print('Total paper = %d' % count)
    print('Remove duplicate = %d' % len(title_map))
    print('No abstract/keyword = %d' % no_keyword_abstract)

    # export the filtered data to file
    with open(export_path[data_name], 'w') as json_file:
        json_file.write(json.dumps(title_map.values()))
