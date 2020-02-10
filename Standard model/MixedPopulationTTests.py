#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:44:48 2019

carry out t-tests on action probabilities, FMF values and MB advantages for the
two devaluation levels on the mixed population simulations

@author: francois
"""

from scipy import stats
import numpy as np
from Welch import welch_ttest

Distribution_name = 'Uniform'

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
    filename = 'Variable ITIs/' + Distribution_name + ' distribution/Mixed population simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    
    
    GoLProbability[:,iti] = np.mean(data['Distribution'][:,1], axis = 0)
    GoLFMFValue[:,iti] = np.mean(data['state1_FMFValue'][:,1], axis = 0)
    GoLMBAdvantage[:,iti] = np.mean(data['state1_MBAdvantage'][:,1], axis = 0)
    
    GoMProbability[:,iti] = np.mean(data['Distribution'][:,2], axis = 0)
    GoMFMFValue[:,iti] = np.mean(data['state1_FMFValue'][:,2], axis = 0)
    GoMMBAdvantage[:,iti] = np.mean(data['state1_MBAdvantage'][:,2], axis = 0)

#testing for normality
W,p_shapiro1 = stats.shapiro(GoLProbability[:,0])
W,p_shapiro2 = stats.shapiro(GoLProbability[:,1]) # p = 0.0053
#testing for homogeneity of variance
chi, p_bartlett1 = stats.bartlett(GoLProbability[:,0], GoLProbability[:,1]) # p=0.0034
# we use Welch's t-test because of the violation of the homogeneity of variance hypothesis
#t_GoLProbability, p_GoLProbability = stats.ttest_ind(GoLProbability[:,0], GoLProbability[:,1])
welch_ttest(GoLProbability[:,0], GoLProbability[:,1])

#testing for normality
W,p_shapiro3 = stats.shapiro(GoLFMFValue[:,0])
W,p_shapiro4 = stats.shapiro(GoLFMFValue[:,1])
#testing for homogeneity of variance
chi, p_bartlett2 = stats.bartlett(GoLFMFValue[:,0], GoLFMFValue[:,1]) #p=1 10^-6
# we use Welch's t-test
#t_GoLFMFValue, p_GoLFMFValue = stats.ttest_ind(GoLFMFValue[:,0], GoLFMFValue[:,1])
welch_ttest(GoLFMFValue[:,0], GoLFMFValue[:,1])

#testing for normality
W,p_shapiro5 = stats.shapiro(GoLMBAdvantage[:,0])
W,p_shapiro6 = stats.shapiro(GoLMBAdvantage[:,1]) # p =1.10^-5
#testing for homogeneity of variance
chi, p_bartlett3 = stats.bartlett(GoLMBAdvantage[:,0], GoLMBAdvantage[:,1])
# we use Student's t-test
t_GoLMBAdvantage, p_GoLMBAdvantage = stats.ttest_ind(GoLMBAdvantage[:,0], GoLMBAdvantage[:,1])
welch_ttest(GoLMBAdvantage[:,0], GoLMBAdvantage[:,1])


#testing for normality
W,p_shapiro7 = stats.shapiro(GoMProbability[:,0]) #p=0.039
W,p_shapiro8 = stats.shapiro(GoMProbability[:,1]) #p=0.0008
#testing for homogeneity of variance
chi, p_bartlett4 = stats.bartlett(GoMProbability[:,0], GoMProbability[:,1]) #p=0.003
# we use Welch's t-test
#t_GoMProbability, p_GoMProbability = stats.ttest_ind(GoMProbability[:,0], GoMProbability[:,1])
welch_ttest(GoMProbability[:,0], GoMProbability[:,1])

#testing for normality
W,p_shapiro9 = stats.shapiro(GoMFMFValue[:,0])
W,p_shapiro10 = stats.shapiro(GoMFMFValue[:,1]) #p=0.0038
#testing for homogeneity of variance
chi, p_bartlett5 = stats.bartlett(GoMFMFValue[:,0], GoMFMFValue[:,1]) #p=1.10-7
# we use Welch's t-test
t_GoMFMFValue, p_GoMFMFValue = stats.ttest_ind(GoMFMFValue[:,0], GoMFMFValue[:,1])
welch_ttest(GoMFMFValue[:,0], GoMFMFValue[:,1])

#testing for normality
W,p_shapiro11 = stats.shapiro(GoMMBAdvantage[:,0]) #p=1.10^-8
W,p_shapiro12 = stats.shapiro(GoMMBAdvantage[:,1])
#testing for homogeneity of variance
chi, p_bartlett6 = stats.bartlett(GoMMBAdvantage[:,0], GoMMBAdvantage[:,1]) #p=5.10-55
# we use Welch's t-test
t_GoMMBAdvantage, p_GoMMBAdvantage = stats.ttest_ind(GoMMBAdvantage[:,0], GoMMBAdvantage[:,1])
welch_ttest(GoMMBAdvantage[:,0], GoMMBAdvantage[:,1])