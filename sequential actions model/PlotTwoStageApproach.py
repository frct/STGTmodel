#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:43:17 2018

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

phenotype = ['ST']

iti_scales = 1
n_blocks = 8
n_trials = 50
x = range(1, 3)

filename = phenotype[0] + 'Simulations with intertrial ' + str(iti_scales) + '.npz'
data = np.load(filename)

a = np.mean(data['goL_counter1'] / n_trials, axis = 0)
a_err = sem(data['goL_counter1'] / n_trials)
b = np.mean(data['goL_counter2'] / n_trials, axis = 0)
b_err = sem(data['goL_counter2'] / n_trials)
y1 = np.stack((a, b), axis = 1)
y1_err = np.stack((a_err, b_err), axis = 1)

a = np.mean(data['goM_counter1'] / n_trials, axis = 0)
a_err = sem(data['goM_counter1'] / n_trials)
b = np.mean(data['goM_counter2'] / n_trials, axis = 0)
b_err = sem(data['goM_counter2'] / n_trials)
y2 = np.stack((a, b), axis = 1)
y2_err = np.stack((a_err, b_err), axis = 1)

a = np.mean(data['exp_counter1'] / n_trials, axis = 0)
a_err = sem(data['exp_counter1'] / n_trials)
b = np.mean(data['exp_counter2'] / n_trials, axis = 0)
b_err = sem(data['exp_counter2'] / n_trials)
y3 = np.stack((a, b), axis = 1)
y3_err = np.stack((a_err, b_err), axis = 1)

for block in range(n_blocks):
    plt.figure(1)
    plt.errorbar(x, y1[block,:], y1_err[block, :], label='block=' + str(block)) #, color=pheno_colours[counter]
    
    plt.figure(2)
    plt.errorbar(x, y2[block,:], y2_err[block, :], label='block=' + str(block))
    
    plt.figure(3)
    plt.errorbar(x, y3[block,:], y3_err[block, :], label='block=' + str(block))
#    y = np.mean(data['goM_counter'] / n_trials, axis = 0)
#    yerr = sem(data['goM_counter'] / n_trials)
#    plt.errorbar(x, y, yerr, label='u_iti=' + str(iti_scale)) #, color=pheno_colours[counter]
#    
#plt.figure(1)
#plt.legend(loc='best')
#plt.savefig('Approach to lever with variable ITI')
#
#plt.figure(2)
#plt.legend(loc='best')
#plt.savefig('Approach to magazine with variable ITI')