#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:53:10 2018

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

phenotypes = ['ST', 'GT'] #, 'IG']
n_phenotypes = len(phenotypes)
pheno_colours = ['r', 'b', 'k']
n_blocks = 8
n_trials = 50
x = range(1, n_blocks + 1)
iti_scale = 1

if iti_scale == 1:
    duration = 'standard'
elif iti_scale == 0.5:
    duration = 'short'
elif iti_scale == 2:
    duration = 'long'


magazine_present = True
lever_present = False

if magazine_present == False:
    ITIcondition = 'magazine absent'
elif lever_present:
    ITIcondition = 'lever present'
else:
    ITIcondition = 'lever absent'
print(ITIcondition)

for counter, phenotype in enumerate(phenotypes):
    #filename = ITIcondition + '/' + phenotype + 'Simulations with intertrial ' + str(iti_scale) + '.npz'
    filename = 'replication Lesaint 2014/' + phenotype + 'Simulations with intertrial ' + str(iti_scale) + '.npz'
    data = np.load(filename)
    
    plt.figure()
    IndividualAverageDA_CS = np.mean(data['dopamineCS'], axis = 0)
    plt.plot(x, IndividualAverageDA_CS)
    
    plt.figure()
    IndividualAverageDA_US = np.mean(data['dopamineUS'], axis = 0)
    plt.plot(x, IndividualAverageDA_US)
    
    plt.figure()
    y = np.mean(IndividualAverageDA_CS, axis = 1)
    yerr = sem(IndividualAverageDA_CS.transpose())
    plt.errorbar(x, y, yerr, color='r', label='CS')
    y = np.mean(IndividualAverageDA_US, axis = 1)
    yerr = sem(IndividualAverageDA_US.transpose())
    plt.errorbar(x, y, yerr, color='b', label='US')
    plt.axis([0, 9, 0, 1])
    plt.legend(loc='best')
    plt.savefig('replication Lesaint 2014//Patterns of Da activity in ' + phenotype + ' for ' + duration + ' intertrial ' + ITIcondition + '.pdf')
    plt.show()