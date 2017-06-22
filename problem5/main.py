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
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score


matplotlib.style.use('ggplot')

if not os.path.exists('data.csv'):
    generate_data()

data = pd.read_csv('data.csv')
em = EM(2, max_iters=1e4)
means = em.fit(data[['x', 'y']].values)

new_data = pd.DataFrame(columns=['x', 'y', 'target'])
for row in data.iterrows():
    x, y = row[1].x, row[1].y
    prob = em.predict([x, y])
    new_data = new_data.append(
        {'x': x, 'y': y, 'target': np.argmax(prob) + 1}, ignore_index=True)

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_aspect('equal')
ax1 = sns.kdeplot(data[data.target == 1].x, data[
    data.target == 1].y, shade=False, kernel='gau')
ax1 = sns.kdeplot(data[data.target == 2].x, data[
    data.target == 2].y, shade=False, kernel='gau')
fig1.savefig('dist_actual.png')

fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_aspect('equal')
ax2 = sns.kdeplot(new_data[new_data.target == 1].x, new_data[
    new_data.target == 1].y, shade=False, shade_lowest=False,
    kernel='gau')
ax2 = sns.kdeplot(new_data[new_data.target == 2].x, new_data[
    new_data.target == 2].y, shade=False, shade_lowest=False,
    kernel='gau')
fig2.savefig('dist_predicted.png')

plt_colors = sns.husl_palette(2)

sns.pairplot(data, x_vars='x', y_vars='y', hue='target', kind='scatter',
             palette={1: plt_colors[0], 2: plt_colors[1]}, size=8)
plt.savefig('data_points_actual.png')


sns.pairplot(new_data, x_vars='x', y_vars='y', hue='target',
             kind='scatter', palette={1: plt_colors[0], 2: plt_colors[1]},
             size=8)
plt.savefig('data_points_predicted.png')

# plot means
m1 = np.array(means[0])
m2 = np.array(means[1])
xs = np.arange(len(means[0]))

fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3 = sns.regplot(xs, m1, fit_reg=False)
fig3.savefig('mean_1_variations.png')

fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4 = sns.regplot(xs, m2, fit_reg=False)
fig4.savefig('mean_2_variations.png')

print('iterations : {}'.format(len(xs)))
print('class 1 estimated mean : {}'.format(m1[-1]))
print('class 2 estimated mean : {}'.format(m2[-1]))

print('adjusted_mutual_info_score: {:.3f}'.format(
    adjusted_mutual_info_score(data.target.values, new_data.target.values)))
