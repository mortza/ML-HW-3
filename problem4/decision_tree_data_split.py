#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13  02:35:51 2017

@author: mortza
"""
import zipfile
import os
import numpy as np
import pandas as pd


zip_ref = zipfile.ZipFile('../adult-census-income.zip', mode='r')
zip_ref.extract('adult.csv', path='./data/original/')

data = pd.read_csv('./data/original/adult.csv')
split_data = np.array_split(data, 10)

os.mkdir('./data/split')

for (index, spl) in enumerate(split_data):
    spl.to_csv('./data/split/data{}.csv'.format(index), index=False, mode='w+')
