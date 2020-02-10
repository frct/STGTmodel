#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:23:32 2018

plot lever and food cup approach under and after flupenthixol injection

@author: francois
"""


import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

n_blocks = 8
n_trials = 50
flupenthixol = 0.9
x = range(1, n_blocks)

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

#filename = 'flupenthixol inhibition/ST flupenthixol inhibition = ' + str(flupenthixol) + '.npz'
filename = 'replication Lesaint 2014/ST flupenthixol inhibition = ' + str(flupenthixol) + '.npz'
data_flu = np.load(filename)

filename = 'replication Lesaint 2014/STSimulations with intertrial ' + str(iti_scale) + '.npz'
data_control = np.load(filename)

plt.figure()
y1 = np.mean(data_flu['goL_counter'] / n_trials, axis = 0)
y1err = sem(data_flu['goL_counter'] / n_trials)
plt.errorbar(x, y1[0:7], y1err[0:7], color='r', label='flu')
y2 = np.mean(data_control['goL_counter'] / n_trials, axis = 0)
y2err = sem(data_control['goL_counter'] / n_trials)
plt.errorbar(x, y2[0:7], y2err[0:7], color='k', label='veh')
plt.axis([0, 9, 0, 1])
plt.legend(loc='best')

plt.savefig('replication Lesaint 2014/ST Approach to lever under flupenthixol treatment.png')

plt.figure()
plt.bar([7.75, 8.25], [y2[7], y1[7]], align='center', width = 0.4, yerr = [y2err[7], y1err[7]], color=['k', 'r'])
plt.savefig('replication Lesaint 2014/ST Approach to lever after flupenthixol treatment.png')
   
filename = 'replication Lesaint 2014/GT flupenthixol inhibition = ' + str(flupenthixol) + '.npz'
data_flu = np.load(filename)

filename = 'replication Lesaint 2014/GTSimulations with intertrial ' + str(iti_scale) + '.npz'
data_control = np.load(filename)
 
plt.figure()
y1 = np.mean(data_flu['goM_counter'] / n_trials, axis = 0)
y1err = sem(data_flu['goM_counter'] / n_trials)
plt.errorbar(x, y1[0:7], y1err[0:7], color='b', label='flu')
y2 = np.mean(data_control['goM_counter'] / n_trials, axis = 0)
y2err = sem(data_control['goM_counter'] / n_trials)
plt.errorbar(x, y2[0:7], y2err[0:7], color='k', label='veh')
plt.axis([0, 9, 0, 1])
plt.legend(loc='best')

plt.savefig('replication Lesaint 2014/GT Approach to food cup under flupenthixol treatment.png')

plt.figure()
plt.bar([7.75, 8.25], [y2[7], y1[7]], align='center', width = 0.4, yerr = [y2err[7], y1err[7]], color=['k', 'b'])
plt.savefig('replication Lesaint 2014/GT Approach to lever after flupenthixol treatment.png')