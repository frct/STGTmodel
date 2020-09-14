# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:39:59 2020

@author: franc
"""


import scipy.io
import numpy as np

mat = scipy.io.loadmat('Compiled_STGT_model_data.mat')
data = mat['Compiled_STGT_mdl_data_PCA']



new_data = data[data[:,3] != 0]

n_trials = new_data.shape[0]

for t in range(n_trials):
    if new_data[t,3] < 0: # goal-tracking predominates
        new_data[t,3] = int(2)
    else:
        new_data[t,3] = int(1)

np.save('Binary classification of data', new_data)
scipy.io.savemat('Binary classification of data.mat', {'new_data': new_data})