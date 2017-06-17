#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17  12:58:48 2017

@author: mortza
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')


def generate_data(t1=400, t2=400, loc1=1, loc2=5, scale1=1.5, scale2=3):
    data = pd.DataFrame(columns=['x', 'y', 'target'])
    for i in range(t1):
        data = data.append({
            'x': np.random.normal(loc1, scale1),
            'y': np.random.normal(loc1, scale1),
            'target': 1}, ignore_index=True)

    for i in range(t2):
        data = data.append({
            'x': np.random.normal(loc2, scale2),
            'y': np.random.normal(loc2, scale2),
            'target': 2}, ignore_index=True)

    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv('data.csv', mode='w+', index=False)
    plt.scatter(data[data.target == 1].x,
                data[data.target == 1].y)
    plt.scatter(data[data.target == 2].x,
                data[data.target == 2].y)
    plt.show()
