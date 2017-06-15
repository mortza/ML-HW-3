#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15  18:14:23 2017

@author: mortza
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.style.use('seaborn')

unpruned_tree_results = pd.read_csv('results_unpruned.csv')
pruned_tree_results = pd.read_csv('results_pruned.csv')


# plot Percent_incorrect vs Number_of_training_instances
num_train_instances = unpruned_tree_results.Number_of_training_instances.values
pruned_percent_incorrect = np.concatenate(
    (pruned_tree_results.Percent_incorrect.values, [0]), axis=0)
upruned_percent_incorrect = unpruned_tree_results.Percent_incorrect.values

plt.figure(1)
plt.plot(num_train_instances, pruned_percent_incorrect, 'o-',
         label='pruned tree')
plt.plot(num_train_instances, upruned_percent_incorrect, 'o-',
         label='unpruned tree')
plt.legend(loc='upper right')
plt.xlim([0, num_train_instances[0] + 1000])
# plt.ylim([0, 25])
plt.xlabel('Number of training instances')
plt.ylabel('incorrect percent')
plt.savefig('fig1.png')

# plot measureTreeSize vs Number_of_training_instances
pruned_tree_size = np.concatenate(
    (pruned_tree_results.measureTreeSize.values, [0]), axis=0)
unpruned_tree_size = unpruned_tree_results.measureTreeSize.values

plt.figure(2)
plt.plot(num_train_instances, pruned_tree_size, 'o-',
         label='pruned tree')
plt.plot(num_train_instances, unpruned_tree_size, 'o-',
         label='unpruned tree')
plt.legend(loc='upper right')
plt.xlim([0, num_train_instances[0] + 1000])
# plt.ylim([0, 25])
plt.xlabel('Number of training instances')
plt.ylabel('Tree Size')
plt.savefig('fig2.png')

# plot measureNumLeaves vs Number_of_training_instances
pruned_num_leaves = np.concatenate(
    (pruned_tree_results.measureNumLeaves.values, [0]), axis=0)
unpruned_num_leaves = unpruned_tree_results.measureNumLeaves.values

plt.figure(3)
plt.plot(num_train_instances, pruned_num_leaves, 'o-',
         label='pruned tree')
plt.plot(num_train_instances, unpruned_num_leaves, 'o-',
         label='unpruned tree')
plt.legend(loc='upper right')
plt.xlim([0, num_train_instances[0] + 1000])
# plt.ylim([0, 25])
plt.xlabel('Number of training instances')
plt.ylabel('Numer of Leaves')
plt.savefig('fig3.png')
