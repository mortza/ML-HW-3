#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15  18:14:23 2017

@author: mortza
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.style.use('seaborn')

result = pd.read_csv('result.csv')

# plot percent_incorrect vs dataset_id
plt.figure(1)
plt.plot(result.dataset_id, result.percent_incorrect, 'o-')
plt.xlabel('dataset id')
plt.ylabel('incorrect classification percent')
plt.savefig('fig1.png')
