#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 18:09:44 2019

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from SEM import sem

Distribution_name = 'Beta distribution'

u_iti = 0.02
n_rats = 20
n_trials = 25
#x = range(1, 3)

width = 0.2
epsilon = 0.1
x = [1-(width + epsilon)/2, 1 + (width + epsilon)/2]

filename = Distribution_name + '/Simulations with intertrial ' + str(u_iti) + '.npz'
data = np.load(filename)

a = np.mean(data['goL_counter1'] / n_trials, axis = 1)
b = np.mean(data['goL_counter2'] / n_trials, axis = 1)
y1 = np.stack((a, b), axis = 1)

a = np.mean(data['goM_counter1'] / n_trials, axis = 1)
b = np.mean(data['goM_counter2'] / n_trials, axis = 1)
y2 = np.stack((a, b), axis = 1)
y2err= sem(y2)

#plt.figure(2)
#plt.boxplot(y2)
#plt.plot(x, np.mean(y2, axis = 0), color = 'black')

#for rat in range(n_rats):
#    plt.figure(1)
#    plt.plot(x, y1[rat,:], color = 'lightgray')
#    
#    plt.figure(2)
#    plt.plot(x, y2[rat,:], color = 'lightgray', marker = '.')
#    
#t_goL, p_goL = stats.ttest_rel(y1[:,0], y1[:,1])
#t_goM, p_goM = stats.ttest_rel(y2[:,0], y2[:,1])



plt.figure(2)
plt.bar(x, np.mean(y2, axis = 0), yerr = y2err, width=width, edgecolor = 'k', facecolor= 'White')
plt.axis([0.5, 3, 0, 1])

for rat in range(n_rats):
    
    plt.figure(2)
    plt.plot(x, y2[rat,:], color = 'lightgray', marker = ',', alpha=1, linewidth=0.5)
plt.xticks(x)

plt.savefig(Distribution_name + '/Probability of approach to food cup during CS period.eps')

W, p = stats.wilcoxon(y2[:,0], y2[:,1])