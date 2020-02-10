#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:30:27 2018

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

phenotypes = ['ST', 'GT', 'IG']
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
elif iti_scale == 5:
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
    #filename = 'magazine ' + magazine + '/' + phenotype + 'Simulations.npz'
    #filename = phenotype + 'Simulations.npz'
    data = np.load(filename)
    
    plt.figure(counter)
    plt.plot(x, data['goL_counter'].transpose() / n_trials)
    
    plt.figure(counter + n_phenotypes)
    plt.plot(x, data['goM_counter'].transpose() / n_trials)
    
    plt.figure(10)
    y = np.mean(data['goL_counter'] / n_trials, axis = 0)
    yerr = sem(data['goL_counter'] / n_trials)
    plt.errorbar(x, y, yerr, color=pheno_colours[counter], label=phenotype)
    
    plt.figure(11)
    y = np.mean(data['goM_counter'] / n_trials, axis = 0)
    yerr = sem(data['goM_counter'] / n_trials)
    plt.errorbar(x, y, yerr, color=pheno_colours[counter], label=phenotype)
    
    

plt.figure(10)
plt.axis([0, 9, 0, 1])
plt.savefig('replication Lesaint 2014/' + 'Approach to lever when ' + ITIcondition + ' during ' + duration + ' ITI.png')
plt.legend(loc='best')
plt.figure(11)
plt.axis([0, 9, 0, 1])
plt.savefig('replication Lesaint 2014/' + 'Approach to magazine when ' + ITIcondition + ' during ' + duration + ' ITI.png')
plt.legend(loc='best')


plt.show()