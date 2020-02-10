#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:51:18 2018

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from SEM import sem

phenotypes = ['ST', 'GT', 'IG']
n_phenotypes = len(phenotypes)
doses = [0, 0.2, 0.4, 0.8]
n_doses = len(doses)
n_rats = 14
n_trials = 50
goL = np.zeros([n_rats * n_phenotypes, n_doses])
goM = np.zeros([n_rats * n_phenotypes, n_doses])

for dose_counter, dose in enumerate(doses):

    for phenotype_counter, phenotype in enumerate(phenotypes):
        filename = 'replication Lesaint 2014/Local flupenthixol administrations/' + phenotype + ' flupenthixol inhibition = ' + str(dose) + '.npz'
        data_flu = np.load(filename)
        goL[phenotype_counter * n_rats : (phenotype_counter + 1) * n_rats, dose_counter] = data_flu['goL_counter'][:,7] / n_trials
        goM[phenotype_counter * n_rats : (phenotype_counter + 1) * n_rats, dose_counter] = data_flu['goM_counter'][:,7] / n_trials
    
goL_avg = np.mean(goL, axis = 0)
goL_sem = sem(goL)
goM_avg = np.mean(goM, axis = 0)
goM_sem = sem(goM)
 
plt.figure()
plt.bar(doses, goL_avg, width = 0.1, align='center', yerr = goL_sem)
plt.savefig('replication Lesaint 2014/Local flupenthixol administrations/Approach to lever.png')

plt.figure()
plt.bar(doses, goM_avg, width = 0.1, align='center', yerr = goM_sem)
plt.savefig('replication Lesaint 2014/Local flupenthixol administrations/Approach to food cup.png')