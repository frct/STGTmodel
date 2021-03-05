# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:47:03 2021

@author: franc
"""


import numpy as np

#folder_path = '4 parameters sessions 4-5'
n_rats = 20
n_parameters = 1

best_params = np.zeros([n_rats, n_parameters + 2])

#rats = np.array([1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19])

for rat in range(1, n_rats + 1):
    params = np.load('Rat' + str(rat) + ' full model optimised parameters.npy')
    optimisation = np.transpose(np.load('Rat' + str(rat) + ' full model optimisation output.npy'))
    
    # converged_optimisation = optimisation[optimisation[:,0] == 1, :]
    # converged_parameters = params[optimisation[:,0] == 1, :]
    # best_idx = np.argmin(converged_optimisation[:,1])
    
    # best_params[rat,:] = np.append(converged_parameters[best_idx,:], converged_optimisation[best_idx, 1:])
    
    best_idx = np.argmin(optimisation[:,1])
    best_params[rat-1, :] = np.append(params[best_idx, :], optimisation[best_idx, 1:])
np.save('Best parameters', best_params )