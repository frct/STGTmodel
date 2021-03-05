# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:07:49 2021

@author: franc
"""


import numpy as np
import ModelOptimisation as mo

data = np.load('Binary classification of data.npy')
data = data[data[:,1]>3] # optimise on sessions 4 and 5 only

rats = np.unique(data[:,0])
#rats = np.array([1])
n_rats = len(rats)

initialisation_points = np.array([[0.2, 0.5, 0.8], [1, 2, 5], [0.05, 0.2, 0.5], [0.2, 0.5, 0.8]])
n_parameters = initialisation_points.shape[0]
n_initialisations = 3**n_parameters
combination_of_initialisations = mo.InitialisationIndices(n_parameters)

for rat in rats:
    rat = int(rat)
    rat_data = data[data[:,0]==rat, 3]
    n_trials = len(rat_data)
    
    optimised_parameters = np.zeros((n_initialisations, n_parameters))
    test = np.zeros((n_initialisations))
    negLL = np.zeros((n_initialisations))
    avg_likelihood = np.zeros((n_initialisations))
    
    for i in range(n_initialisations):
        x0 = np.zeros((n_parameters))
        for p in range(n_parameters):
            x0[p] = initialisation_points[p, combination_of_initialisations[i, p]]
        res = mo.OptimizeParameters(x0, rat_data)
        optimised_parameters[i,:] = res.x
        test[i] = res.success
        negLL[i] = res.fun
        avg_likelihood[i] = np.exp(-res.fun / n_trials)
    
    np.save('Rat' + str(rat) + ' full model optimised parameters', optimised_parameters)  
    np.save('Rat' + str(rat) + ' full model optimisation output', (test, negLL, avg_likelihood))