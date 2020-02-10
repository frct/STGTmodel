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

ITI_colours = ['SkyBlue', 'IndianRed'] #'Plum', 
ITI_labels = ['Short ITI', 'Long ITI'] #, 'Intermediate ITI'

u_itis = np.array([0.01, 0.2])
n_blocks = 10
n_trials = 25
x = range(1, n_blocks + 1)

AverageDistribution = np.zeros((3,2))
AverageDistribution_err = np.zeros((3,2))
AverageFMFvalues = np.zeros((3,2))
AverageFMFvalues_err = np.zeros((3,2))
AverageAdvantage = np.zeros((3,2))
AverageAdvantage_err = np.zeros((3,2))

for iti, u_iti in enumerate(u_itis):
    filename = 'Variable ITIs/' + phenotype + 'Simulations with intertrial ' + str(iti) + '.npz'
    #filename = phenotype + 'Simulations.npz'
    data = np.load(filename)
    
    plt.figure(1)
    y = np.mean(data['goL_counter'] / n_trials, axis = 0)
    yerr = sem(data['goL_counter'] / n_trials)
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], marker = 'o', color=ITI_colours[iti])
    
    plt.figure(2)
    y = np.mean(data['goM_counter'] / n_trials, axis = 0)
    yerr = sem(data['goM_counter'] / n_trials)
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], marker='o', color=ITI_colours[iti])
    
    plt.figure(3)
    IndividualAverageDA_CS = np.mean(data['dopamineCS'], axis = 0)
    y = np.mean(IndividualAverageDA_CS, axis = 1)
    yerr = sem(IndividualAverageDA_CS.transpose())
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], marker='o', color=ITI_colours[iti])
    
    plt.figure(4)
    IndividualAverageDA_US = np.mean(data['dopamineUS'], axis = 0)
    y = np.mean(IndividualAverageDA_US, axis = 1)
    yerr = sem(IndividualAverageDA_US.transpose())
    plt.errorbar(x, y, yerr, label=ITI_labels[iti], marker='o', color=ITI_colours[iti])
    
    
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
plt.axis([0, 11, 0, 1])
plt.savefig('Variable ITIs/'  + phenotype + ' approach to lever for different ITI durations.eps')

plt.figure(2)
plt.legend(loc='best')
plt.axis([0, 11, 0, 1])
plt.savefig('Variable ITIs/'  + phenotype + ' approach to magazine for different ITI durations.eps')

plt.figure(3)
plt.legend(loc='best')
plt.axis([0, 11, 0, 1])
plt.savefig('Variable ITIs/'  + phenotype + 'CS Dopamine activity for different ITI durations.eps')

plt.figure(4)
plt.legend(loc='best')
plt.axis([0, 11, 0, 1])
plt.savefig('Variable ITIs/'  + phenotype + 'US Dopamine activity for different ITI durations.eps')


ind = np.arange(3)  # the x locations for the groups
width = 0.2  # the width of the bars
epsilon = 0.1

fig, ax = plt.subplots()
rects1 = ax.bar(ind - (width+epsilon)/2, AverageDistribution[:,0], width, yerr=AverageDistribution_err[:,0],
                color='SkyBlue', label='Short ITI')
#rects2 = ax.bar(ind, AverageDistribution[:,1], width, yerr=AverageDistribution_err[:,1],
#                color='Plum', label='Intermediate ITI')
rects3 = ax.bar(ind + (width+epsilon)/2, AverageDistribution[:,1], width, yerr=AverageDistribution_err[:,1],
                color='IndianRed', label='Long ITI')
ax.set_xticks(ind)
ax.set_xticklabels(('explore', 'goL', 'goM'))
ax.legend()
plt.savefig('Variable ITIs/'  + phenotype + ' action probability distribution.eps')

fig2, ax2 = plt.subplots()
rects1 = ax2.bar(ind - (width+epsilon)/2, AverageFMFvalues[:,0], width, yerr=AverageFMFvalues_err[:,0],
                color='SkyBlue', label='Short ITI')
#rects2 = ax2.bar(ind, AverageFMFvalues[:,1], width, yerr=AverageFMFvalues_err[:,1],
#                color='Plum', label='Intermediate ITI')
rects3 = ax2.bar(ind + (width+epsilon)/2, AverageFMFvalues[:,1], width, yerr=AverageFMFvalues_err[:,1],
                color='IndianRed', label='Long ITI')
rects4 = ax2.bar(ind - (width+epsilon)/2, AverageAdvantage[:,0], width, yerr=AverageAdvantage_err[:,0],
                color='SkyBlue')
#rects5 = ax2.bar(ind, AverageAdvantage[:,1], width, yerr=AverageAdvantage_err[:,1],
#                color='Plum')
rects6 = ax2.bar(ind + (width+epsilon)/2, AverageAdvantage[:,1], width, yerr=AverageAdvantage_err[:,1],
                color='IndianRed')

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
ax2.axhline(0, color='black', lw=0.5)
ax2.set_xticks(ind)
ax2.set_xticklabels(('explore', 'goL', 'goM'))
ax2.legend()
plt.savefig('Variable ITIs/'  + phenotype + ' action advantage and value.eps')