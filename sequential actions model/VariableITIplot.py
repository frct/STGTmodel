#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:18:05 2018

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

phenotype = 'ST'

magazine_present = True
lever_present = False

if magazine_present == False:
    ITIcondition = 'magazine absent'
elif lever_present:
    ITIcondition = 'lever present'
else:
    ITIcondition = 'lever absent'

iti_scales = [0.5, 1, 2]
print(iti_scales)
n_blocks = 8
n_trials = 50
x = range(1, n_blocks + 1)

for counter, iti_scale in enumerate(iti_scales):
    filename = ITIcondition + '/' + phenotype + 'Simulations with intertrial ' + str(iti_scale) + '.npz'
    #filename = phenotype + 'Simulations.npz'
    data = np.load(filename)
    
    plt.figure(1)
    y = np.mean(data['goL_counter'] / n_trials, axis = 0)
    yerr = sem(data['goL_counter'] / n_trials)
    plt.errorbar(x, y, yerr, label='u_iti=' + str(iti_scale)) #, color=pheno_colours[counter]
    
    plt.figure(2)
    y = np.mean(data['goM_counter'] / n_trials, axis = 0)
    yerr = sem(data['goM_counter'] / n_trials)
    plt.errorbar(x, y, yerr, label='u_iti=' + str(iti_scale)) #, color=pheno_colours[counter]
    
    plt.figure(3)
    IndividualAverageDA_CS = np.mean(data['dopamineCS'], axis = 0)
    y = np.mean(IndividualAverageDA_CS, axis = 1)
    yerr = sem(IndividualAverageDA_CS.transpose())
    plt.errorbar(x, y, yerr, label='u_iti=' + str(iti_scale))
    
    plt.figure(4)
    IndividualAverageDA_US = np.mean(data['dopamineUS'], axis = 0)
    y = np.mean(IndividualAverageDA_US, axis = 1)
    yerr = sem(IndividualAverageDA_US.transpose())
    plt.errorbar(x, y, yerr, label='u_iti=' + str(iti_scale))

    
plt.figure(1)
plt.legend(loc='best')
plt.axis([0, 9, 0, 1])
plt.savefig(ITIcondition + '/' +phenotype + ' approach to lever for different ITI durations')

plt.figure(2)
plt.legend(loc='best')
plt.axis([0, 9, 0, 1])
plt.savefig(ITIcondition + '/' +phenotype + ' approach to magazine for different ITI durations')

plt.figure(3)
plt.legend(loc='best')
plt.axis([0, 9, 0, 1])
plt.savefig(ITIcondition + '/' +phenotype + 'CS Dopamine activity for different ITI durations')

plt.figure(4)
plt.legend(loc='best')
plt.axis([0, 9, 0, 1])
plt.savefig(ITIcondition + '/' +phenotype + 'US Dopamine activity for different ITI durations')