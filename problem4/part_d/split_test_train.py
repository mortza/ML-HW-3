#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15  22:43:56 2017

@author: mortza
"""
import pandas as pd

data = pd.read_csv('../data/original/adult.csv')
data = data.sample(frac=1).reset_index(drop=True)
test = data.sample(frac=0.1)
train = data.sample(frac=0.9)
test.to_csv('../data/split/part_d_test.csv', index=False)
train.to_csv('../data/split/part_d_train.csv', index=False)
