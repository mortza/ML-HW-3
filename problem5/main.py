#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16  03:22:59 2017

@author: mortza
"""
import numpy as np
import pandas as pd
import os
from gen_data import generate_data
from core.EM import EM
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

if not os.path.exists('data.csv'):
    generate_data()

data = pd.read_csv('data.csv')
em = EM(2)
means = em.fit(data[['x', 'y']].values)

new_data = pd.DataFrame(columns=['x', 'y', 'target'])
for row in data.iterrows():
    x, y = row[1].x, row[1].y
    prob = em.predict([x, y])
    new_data = new_data.append(
        {'x': x, 'y': y, 'target': np.argmax(prob) + 1}, ignore_index=True)


fig, (ax1, ax2) = plt.subplots(2, sharey=True)

ax1.scatter(data[data.target == 1].x, data[data.target == 1].y)
ax1.scatter(data[data.target == 2].x, data[data.target == 2].y)
ax1.set_title('original data')

ax2.scatter(new_data[new_data.target == 1].x, new_data[
            new_data.target == 1].y)
ax2.scatter(new_data[new_data.target == 2].x, new_data[
            new_data.target == 2].y)
print(len(new_data))
ax2.set_title('predicted data')
plt.show()

# # plot means
# m1 = np.ones((len(means)))
# m2 = np.ones((len(means)))
# xs = np.arange(len(means)) + 1

# for (i, k) in enumerate(means):
#     m1[i] = k[0]
#     m2[i] = k[1]

# plt.plot(xs, m1)
# plt.plot(xs, m2)
# print(m1)
# plt.show()
