# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:33:07 2020

@author: franc
"""

import numpy as np

folder_path = 'two actions/no gamma two actions/Q=0.5 normal initialisation'
n_rats = 20
n_parameters = 4

best_params = np.zeros([n_rats, n_parameters + 1])


for rat in range(n_rats):
    params = np.load(folder_path + '/Rat' + str(rat+1) + ' parameters.npy')
    negLL = np.load(folder_path + '/Rat' + str(rat+1) + ' negLL.npy')
    best_idx = np.argmin(negLL)
    
    best_params[rat,:] = np.append(params[best_idx,:], negLL[best_idx])
    
np.save('Best parameters', best_params )