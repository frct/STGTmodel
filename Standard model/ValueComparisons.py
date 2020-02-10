#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:34:30 2019

compare v(M) and v(L)

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from SEM import sem

Distribution_name = 'Beta'
ITI_colours = ['SkyBlue', 'IndianRed']
ITI_labels = ['Short ITI', 'Long ITI']

u_itis = np.array([0.01, 0.1])
n_blocks = 60
n_trials = 25
x = range(1, n_blocks + 1)

for iti, u_iti in enumerate(u_itis):
    
    filename = 'Variable ITIs/' + Distribution_name + ' distribution/Mixed population long simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    a = data['state1_FMFValue']
    V_L = np.mean(a[:,1,:], axis = 1).reshape((n_blocks, n_trials)).transpose()
    V_M = np.mean(a[:,2,:], axis = 1)
    
    plt.figure(iti)
    yerr= sem(V_L).transpose()
    y = np.mean(V_L, axis = 0)
    plt.errorbar(x, np.mean(V_L, axis = 0), color=ITI_colours[iti], marker = 'o', label='Lever')
    plt.errorbar(x, np.mean(V_M, axis = 0), color=ITI_colours[iti], marker = '+', label='Food cup')
    plt.legend()
    plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/' + ITI_labels[iti] + ' feature values.eps')