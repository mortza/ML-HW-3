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

if not os.path.exists('data.csv'):
    generate_data()

data = pd.read_csv('data.csv')
em = EM(2)
em.fit(data[['x', 'y']].values[10:30, :])
