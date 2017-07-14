#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
just check how many data instances in the json
"""

import os
import json

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    print(os.getcwd())
    file_path = '../dataset/json/gradle_test_methodnaming.json'
    with open(file_path) as json_string:
        data = json.load(json_string)
        print(len(data))