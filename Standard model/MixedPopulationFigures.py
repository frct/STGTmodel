#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:32:44 2018

plot figures for population mix

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from SEM import sem

Distribution_name = 'Beta'
ITI_colours = ['SkyBlue', 'IndianRed']
ITI_labels = ['Short ITI', 'Long ITI']

u_itis = np.array([0.01, 0.1]) #, 0.1
n_blocks = 60
n_trials = 25
x = range(1, n_blocks + 1)

AverageDistribution = np.zeros((3,3))
AverageDistribution_err = np.zeros((3,3))
AverageFMFvalues = np.zeros((3,3))
AverageFMFvalues_err = np.zeros((3,3))
AverageAdvantage = np.zeros((3,3))
AverageAdvantage_err = np.zeros((3,3))

for iti, u_iti in enumerate(u_itis):
    
    filename = 'Variable ITIs/' + Distribution_name + ' distribution/Mixed population long simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    sio.savemat('goL_counter for ' + ITI_labels[iti] + ' ITI', {'goL': data['goL_counter']})
    sio.savemat('goM_counter for ' + ITI_labels[iti] + ' ITI', {'goM': data['goM_counter']})
    
    plt.figure(1)
    y = np.mean(data['goL_counter'] / n_trials, axis = 0)
    yerr = sem(data['goL_counter'] / n_trials)
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], color=ITI_colours[iti], marker = 'o')
    
    plt.figure(2)
    y = np.mean(data['goM_counter'] / n_trials, axis = 0)
    yerr = sem(data['goM_counter'] / n_trials)
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], color=ITI_colours[iti], marker = 'o')
    
    plt.figure(3)
    IndividualAverageDA_CS = np.mean(data['dopamineCS'], axis = 0)
    y = np.mean(IndividualAverageDA_CS, axis = 1)
    yerr = sem(IndividualAverageDA_CS.transpose())
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], color=ITI_colours[iti], marker = 'o')
    
    plt.figure(4)
    IndividualAverageDA_US = np.mean(data['dopamineUS'], axis = 0)
    y = np.mean(IndividualAverageDA_US, axis = 1)
    yerr = sem(IndividualAverageDA_US.transpose())
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], color=ITI_colours[iti], marker = 'o')
    
    
    IndividualAverageDistribution = np.mean(data['Distribution'], axis = 0)
    AverageDistribution[:,iti] = np.mean(IndividualAverageDistribution, axis = 1)
    AverageDistribution_err[:,iti] = sem(IndividualAverageDistribution.transpose())
    
    IndividualAverageFMF = np.mean(data['state1_FMFValue'], axis = 0)
    AverageFMFvalues[:,iti] = np.mean(IndividualAverageFMF, axis = 1)
    AverageFMFvalues_err[:,iti] = sem(IndividualAverageFMF.transpose())
    
    IndividualAverageAdvantage = np.mean(data['state1_MBAdvantage'], axis = 0)
    AverageAdvantage[:,iti] = np.mean(IndividualAverageAdvantage, axis = 1)
    AverageAdvantage_err[:,iti] = sem(IndividualAverageAdvantage.transpose())
    
plt.figure(1)
plt.legend(loc='best')
plt.axis([0, n_blocks + 1, 0, 1])
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Mixed population approach to lever for different ITI durations.eps')

plt.figure(2)
plt.legend(loc='best')
plt.axis([0, n_blocks + 1, 0, 1])
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Mixed population approach to magazine for different ITI durations.eps')

plt.figure(3)
plt.legend(loc='best')
plt.axis([0, n_blocks + 1, -0.05, 1])
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Mixed population CS Dopamine activity for different ITI durations.eps')

plt.figure(4)
plt.legend(loc='best')
plt.axis([0, n_blocks + 1, 0, 1])
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Mixed population US Dopamine activity for different ITI durations.eps')


ind = np.arange(3)  # the x locations for the actions: explore, goL, and goM
width = 0.2  # the width of the bars
epsilon = 0.1 #gap between grouped bard

fig, ax = plt.subplots()
rects1 = ax.bar(ind - (width + epsilon) / 2, AverageDistribution[:,0], width, yerr=AverageDistribution_err[:,0],
                color='SkyBlue', label='Short ITI')
rects2 = ax.bar(ind + (width + epsilon) / 2, AverageDistribution[:,1], width, yerr=AverageDistribution_err[:,1],
                color='IndianRed', label='Long ITI')
ax.set_xticks(ind)
ax.set_xticklabels(('explore', 'goL', 'goM'))
ax.legend()
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Mixed population action probability distribution.eps')

fig2, ax2 = plt.subplots()
rects1 = ax2.bar(ind - (width + epsilon) / 2, AverageFMFvalues[:,0], width, yerr=AverageFMFvalues_err[:,0],
                color='SkyBlue', label='Short ITI')
rects2 = ax2.bar(ind + (width + epsilon) / 2, AverageFMFvalues[:,1], width, yerr=AverageFMFvalues_err[:,1],
                color='IndianRed', label='Long ITI')
rects4 = ax2.bar(ind - (width + epsilon) / 2, AverageAdvantage[:,0], width, yerr=AverageAdvantage_err[:,0],
                color='SkyBlue')
rects5 = ax2.bar(ind + (width + epsilon) / 2, AverageAdvantage[:,1], width, yerr=AverageAdvantage_err[:,1],
                color='IndianRed')
ax2.axhline(0, color='black', lw=0.5)
ax2.set_xticks(ind)
ax2.set_xticklabels(('explore', 'goL', 'goM'))
ax2.legend()
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Mixed population action advantage and value.eps')