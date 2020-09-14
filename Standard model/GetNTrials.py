# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:29:24 2020

@author: franc
"""


import numpy as np 

data = np.load('Binary classification of data.npy')

rats = np.unique(data[:,0])
n_trials = np.zeros(20)

for index, rat in enumerate(rats):
    rat = int(rat)
    b = data[:,0] == rat
    rat_data = data[b,3]
    n_trials[index] = len(rat_data)