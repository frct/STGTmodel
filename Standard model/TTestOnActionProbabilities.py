#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:17:27 2019

carry out t-tests on action probabilities, FMF values and MB advantages for the
two devaluation levels

@author: francois
"""

from scipy import stats
import numpy as np

phenotype = 'GT'

u_itis = np.array([0.01, 0.1])
n_itis = len(u_itis)
n_rats = 20

GoLProbability = np.zeros((n_rats, n_itis))
GoLFMFValue = np.zeros((n_rats, n_itis))
GoLMBAdvantage = np.zeros((n_rats, n_itis))

GoMProbability = np.zeros((n_rats, n_itis))
GoMFMFValue = np.zeros((n_rats, n_itis))
GoMMBAdvantage = np.zeros((n_rats, n_itis))

for iti, u_iti in enumerate(u_itis):
    filename = 'Variable ITIs/' + phenotype + 'Simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    
    
    GoLProbability[:,iti] = np.mean(data['Distribution'][:,1], axis = 0)
    GoLFMFValue[:,iti] = np.mean(data['state1_FMFValue'][:,1], axis = 0)
    GoLMBAdvantage[:,iti] = np.mean(data['state1_MBAdvantage'][:,1], axis = 0)
    
    GoMProbability[:,iti] = np.mean(data['Distribution'][:,2], axis = 0)
    GoMFMFValue[:,iti] = np.mean(data['state1_FMFValue'][:,2], axis = 0)
    GoMMBAdvantage[:,iti] = np.mean(data['state1_MBAdvantage'][:,2], axis = 0)
    
t_GoLProbability, p_GoLProbability = stats.ttest_ind(GoLProbability[:,0], GoLProbability[:,1])
t_GoLFMFValue, p_GoLFMFValue = stats.ttest_ind(GoLFMFValue[:,0], GoLFMFValue[:,1])
t_GoLMBAdvantage, p_GoLMBAdvantage = stats.ttest_ind(GoLMBAdvantage[:,0], GoLMBAdvantage[:,1])

t_GoMProbability, p_GoMProbability = stats.ttest_ind(GoMProbability[:,0], GoMProbability[:,1])
t_GoMFMFValue, p_GoMFMFValue = stats.ttest_ind(GoMFMFValue[:,0], GoMFMFValue[:,1])
t_GoMMBAdvantage, p_GoMMBAdvantage = stats.ttest_ind(GoMMBAdvantage[:,0], GoMMBAdvantage[:,1])