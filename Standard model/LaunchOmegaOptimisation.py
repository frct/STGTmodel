# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:30:58 2021

@author: franc
"""

import numpy as np
import ModelOptimisation as mo

data = np.load('Binary classification of data.npy')
data = data[data[:,1]>3] # optimise on sessions 4 and 5 only

rats = np.unique(data[:,0])
n_rats = len(rats)

omega = np.zeros((n_rats))
test = np.zeros((n_rats))
negLL = np.zeros((n_rats))
avg_likelihood = np.zeros((n_rats))

for index, rat in enumerate(rats):
    rat = int(rat)
    rat_data = data[data[:,0]==rat, 3]
    n_trials = len(rat_data)
    res = mo.OptimizeOmega(rat_data)
    omega[index] = res.x
    test[index] = res.success
    negLL[index] = res.fun
    avg_likelihood[index] = np.exp(-res.fun / n_trials)
    
np.save('Omega only optimisation results', (omega, test, negLL, avg_likelihood))  