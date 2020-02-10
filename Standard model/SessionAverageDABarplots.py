#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:02:49 2019

barplots of dopamine activity at CS and US, averaged over all sessions and 
averaged for sessions 1-3 and 4-10 separately

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

Distribution_name = 'Beta'
ITI_colours = ['SkyBlue', 'IndianRed']
ITI_labels = ['Short ITI', 'Long ITI']

u_itis = np.array([0.01, 0.1])
n_blocks = 10
n_trials = 25
n_rats = 20
x = range(1, n_blocks + 1)

All_session_average_CS = np.zeros((2,1))
All_session_average_CS_err = np.zeros((2,1))
All_session_average_US = np.zeros((2,1))
All_session_average_US_err = np.zeros((2,1))

Early_sessions_average_CS = np.zeros((2,1))
Early_sessions_average_CS_err = np.zeros((2,1))
Early_sessions_average_US = np.zeros((2,1))
Early_sessions_average_US_err = np.zeros((2,1))

Late_sessions_average_CS = np.zeros((2,1))
Late_sessions_average_CS_err = np.zeros((2,1))
Late_sessions_average_US = np.zeros((2,1))
Late_sessions_average_US_err = np.zeros((2,1))

Rats_early_averageCS = np.zeros((n_rats,2))
Rats_late_averageCS = np.zeros((n_rats,2))
Rats_early_averageUS = np.zeros((n_rats,2))
Rats_late_averageUS = np.zeros((n_rats,2))

for iti, u_iti in enumerate(u_itis):
    
    filename = 'Variable ITIs/' + Distribution_name + ' distribution/Mixed population simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    
    plt.figure(1)
    IndividualAverageDA_CS = np.mean(data['dopamineCS'], axis = 0)
    Session_Average_DA_CS = np.mean(IndividualAverageDA_CS, axis = 1)
    Session_Average_DA_CS_err = sem(IndividualAverageDA_CS.transpose())
    plt.errorbar(x, Session_Average_DA_CS, Session_Average_DA_CS_err, label=ITI_labels[iti], color=ITI_colours[iti], marker = 'o')
    
    Rats_SessionAverage_DA_CS = np.mean(IndividualAverageDA_CS, axis = 0)
    All_session_average_CS[iti] = np.mean(Rats_SessionAverage_DA_CS)
    All_session_average_CS_err[iti] =sem(Rats_SessionAverage_DA_CS)
    
    for rat in range(n_rats):        
        Rats_early_averageCS[rat,iti] = np.mean(IndividualAverageDA_CS[0:3,rat])
        Rats_late_averageCS[rat,iti] = np.mean(IndividualAverageDA_CS[3:,rat])
        
    Early_sessions_average_CS[iti] = np.mean(Rats_early_averageCS[:,iti])
    Early_sessions_average_CS_err[iti] = sem(Rats_early_averageCS[:,iti])

    Late_sessions_average_CS[iti] = np.mean(Rats_late_averageCS[:,iti])
    Late_sessions_average_CS_err[iti] = sem(Rats_late_averageCS[:,iti])
    
    
    plt.figure(2)
    IndividualAverageDA_US = np.mean(data['dopamineUS'], axis = 0)
    Session_Average_DA_US = np.mean(IndividualAverageDA_US, axis = 1)
    Session_Average_DA_US_err = sem(IndividualAverageDA_US.transpose())
    plt.errorbar(x, Session_Average_DA_US, Session_Average_DA_US_err, label=ITI_labels[iti], color=ITI_colours[iti], marker = 'o')
    
    Rats_SessionAverage_DA_US = np.mean(IndividualAverageDA_US, axis = 0)
    All_session_average_US[iti] = np.mean(Rats_SessionAverage_DA_US)
    All_session_average_US_err[iti] =sem(Rats_SessionAverage_DA_US) 
    
    for rat in range(n_rats):        
        Rats_early_averageUS[rat,iti] = np.mean(IndividualAverageDA_US[0:3,rat])
        Rats_late_averageUS[rat,iti] = np.mean(IndividualAverageDA_US[3:,rat])
    
    Early_sessions_average_US[iti] = np.mean(Rats_early_averageUS[:,iti])
    Early_sessions_average_US_err[iti] = sem(Rats_early_averageUS[:,iti])
    
    Late_sessions_average_US[iti] = np.mean(Rats_late_averageUS[:,iti])
    Late_sessions_average_US_err[iti] = sem(Rats_late_averageUS[:,iti])
    
plt.figure(1)
plt.legend(loc='best')
plt.axis([0, n_blocks + 1, 0, 1])
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Evolution of CS da.eps')

plt.figure(2)
plt.legend(loc='best')
plt.axis([0, n_blocks + 1, 0, 1])
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Evolution of US da.eps')



ind = np.arange(2)  # the x locations for the actions: explore, goL, and goM
width = 0.2  # the width of the bars
epsilon = 0.1 #gap between grouped bard

fig, ax = plt.subplots()
rects1 = ax.bar(ind[0] - (width + epsilon) / 2, All_session_average_CS[0], width, 
                color='SkyBlue', label='Short ITI', yerr=All_session_average_CS_err[0])
rects2 = ax.bar(ind[0] + (width + epsilon) / 2, All_session_average_CS[1], width, 
                color='IndianRed', label='Long ITI', yerr=All_session_average_CS_err[1])
rects3 = ax.bar(ind[1] - (width + epsilon) / 2, All_session_average_US[0], width, 
                color='SkyBlue', yerr=All_session_average_US_err[0])
rects4 = ax.bar(ind[1] + (width + epsilon) / 2, All_session_average_US[1], width, 
                color='IndianRed', yerr=All_session_average_US_err[1])

ax.set_xticks(ind)
ax.set_xticklabels(('CS', 'US'))
ax.legend()
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/All sessions average dopamine.eps')

fig2, ax2 = plt.subplots()
rects1 = ax2.bar(ind[0] - (width + epsilon) / 2, Early_sessions_average_CS[0], width,
                color='SkyBlue', label='sessions1-3', yerr=Early_sessions_average_CS_err[0])
rects2 = ax2.bar(ind[0] + (width + epsilon) / 2, Late_sessions_average_CS[0], width,
                color='SteelBlue', label='sessions4-10', yerr=Late_sessions_average_CS_err[0])
rects4 = ax2.bar(ind[1] - (width + epsilon) / 2, Early_sessions_average_US[0], width,
                color='SkyBlue', yerr=Early_sessions_average_US_err[0])
rects5 = ax2.bar(ind[1] + (width + epsilon) / 2, Late_sessions_average_US[0], width,
                color='Steelblue', yerr=Late_sessions_average_US_err[0])
ax2.axhline(0, color='black', lw=0.5)
ax2.set_xticks(ind)
ax2.set_xticklabels(('CS', 'US'))
ax2.legend()
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Short ITI Da average.eps')

fig2, ax2 = plt.subplots()
rects1 = ax2.bar(ind[0] - (width + epsilon) / 2, Early_sessions_average_CS[1], width,
                color='Salmon', label='sessions1-3', yerr=Early_sessions_average_CS_err[1])
rects2 = ax2.bar(ind[0] + (width + epsilon) / 2, Late_sessions_average_CS[1], width,
                color='IndianRed', label='sessions4-10', yerr=Late_sessions_average_CS_err[1])
rects4 = ax2.bar(ind[1] - (width + epsilon) / 2, Early_sessions_average_US[1], width,
                color='Salmon', yerr=Early_sessions_average_US_err[1])
rects5 = ax2.bar(ind[1] + (width + epsilon) / 2, Late_sessions_average_US[1], width,
                color='IndianRed', yerr=Late_sessions_average_US_err[1])
ax2.axhline(0, color='black', lw=0.5)
ax2.set_xticks(ind)
ax2.set_xticklabels(('CS', 'US'))
ax2.legend()
plt.savefig('Variable ITIs/' + Distribution_name + ' distribution/Long ITI Da average.eps')