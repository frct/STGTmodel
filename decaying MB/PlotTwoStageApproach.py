#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:43:17 2018

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem


u_iti = 0.8
n_blocks = 8
n_trials = 50
x = range(1, 3)

#filename = phenotype[0] + 'FMFonly_Simulations with intertrial ' + str(u_iti) + '.npz'
filename = 'Decaying MB Simulations.npz'
data = np.load(filename)

a = np.mean(data['goL_counter1'] / n_trials, axis = 0)
a_err = sem(data['goL_counter1'] / n_trials)
b = np.mean(data['goL_counter2'] / (data['goL_counter2'] + data['goM_counter2']), axis = 0)
b_err = sem(data['goL_counter2'] / (data['goL_counter2'] + data['goM_counter2']))
y1 = np.stack((a, b), axis = 1)
y1_err = np.stack((a_err, b_err), axis = 1)

a = np.mean(data['goM_counter1'] / n_trials, axis = 0)
a_err = sem(data['goM_counter1'] / n_trials)
b = np.mean(data['goM_counter2'] / (data['goL_counter2'] + data['goM_counter2']), axis = 0)
b_err = sem(data['goM_counter2'] / (data['goL_counter2'] + data['goM_counter2']))
y2 = np.stack((a, b), axis = 1)
y2_err = np.stack((a_err, b_err), axis = 1)


for block in range(n_blocks):
    plt.figure(1)
    plt.errorbar(x, y1[block,:], y1_err[block, :], label='block=' + str(block)) #, color=pheno_colours[counter]
    
    plt.figure(2)
    plt.errorbar(x, y2[block,:], y2_err[block, :], label='block=' + str(block))
    

#    y = np.mean(data['goM_counter'] / n_trials, axis = 0)
#    yerr = sem(data['goM_counter'] / n_trials)
#    plt.errorbar(x, y, yerr, label='u_iti=' + str(iti_scale)) #, color=pheno_colours[counter]
#    
plt.figure(1)
plt.legend(loc='best')
#plt.savefig('Approach to lever with variable ITI')
#
plt.figure(2)
plt.legend(loc='best')
#plt.savefig('Approach to magazine with variable ITI')