# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:46:12 2021

@author: franc
"""
import numpy as np

data = np.load('Binary classification of data.npy')
data = data[data[:,1]>3]

n_trials = np.zeros((20))
for rat in range(20):
    n_trials[rat] = len(data[data[:,0] == rat+1, 0])