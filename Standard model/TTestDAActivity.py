#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:46:50 2019

t-tests on dopamine activity

@author: francois
"""

from scipy import stats
from Welch import welch_ttest
import numpy as np

Distribution_name = 'Beta'

u_itis = np.array([0.01, 0.1])
n_itis = len(u_itis)
n_sessions = 10
n_rats = 20
n_early = 3 #number of the sessions considered as early
n_late = n_sessions - n_early

Session_Average_DA_CS = np.zeros((n_rats, n_itis))
Session_Average_DA_US = np.zeros((n_rats, n_itis))
Early_Average_DA_CS = np.zeros((n_rats, n_itis))
Early_Average_DA_US = np.zeros((n_rats, n_itis))
Late_Average_DA_CS = np.zeros((n_rats, n_itis))
Late_Average_DA_US = np.zeros((n_rats, n_itis))
        
for iti, u_iti in enumerate(u_itis):
    filename = 'Variable ITIs/' + Distribution_name + ' distribution/Mixed population simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    
    IndividualAverageDA_CS = np.mean(data['dopamineCS'], axis = 0)
    IndividualAverageDA_US = np.mean(data['dopamineUS'], axis = 0)

    
    for rat in range(n_rats):
        Session_Average_DA_CS[rat, iti] = np.mean(IndividualAverageDA_CS[:,rat])
        Session_Average_DA_US[rat, iti] = np.mean(IndividualAverageDA_US[:,rat])
        Early_Average_DA_CS[rat, iti] = np.mean(IndividualAverageDA_CS[0:3,rat])
        Early_Average_DA_US[rat, iti] = np.mean(IndividualAverageDA_US[0:3,rat])
        Late_Average_DA_CS[rat, iti] = np.mean(IndividualAverageDA_CS[3:,rat])
        Late_Average_DA_US[rat, iti] = np.mean(IndividualAverageDA_US[3:,rat])
        
        # np.mean(IndividualAverageDA_CS, axis = 1)
    

#testing for normality
W,p_shapiro1 = stats.shapiro(Session_Average_DA_CS[:,0])
W,p_shapiro2 = stats.shapiro(Session_Average_DA_CS[:,1])
#testing for homogeneity of variance
chi, p_bartlett1 = stats.bartlett(Session_Average_DA_CS[:,0], Session_Average_DA_CS[:,1]) # p =0.0005
#since there is a violation of variance homogeneity, we resort to Welch's t-test
t_AllSessionsCS, p_AllSessionsCS = stats.ttest_ind(Session_Average_DA_CS[:,0], Session_Average_DA_CS[:,1], equal_var=False)
welch_ttest(Session_Average_DA_CS[:,0], Session_Average_DA_CS[:,1])

#testing for normality
W,p_shapiro3 = stats.shapiro(Session_Average_DA_US[:,0])
W,p_shapiro4 = stats.shapiro(Session_Average_DA_US[:,1]) #found p=0.003
#testing for homogeneity of variance
chi, p_bartlett2 = stats.bartlett(Session_Average_DA_US[:,0], Session_Average_DA_US[:,1]) # p=5.10^-7 
# we use Welch's t-test because of violation of varianve homogeneity, and ignore the test for normality result which only concerns one of the two groups
t_AllSessionsUS, p_AllSessionsUS = stats.ttest_ind(Session_Average_DA_US[:,0], Session_Average_DA_US[:,1], equal_var=False)
welch_ttest(Session_Average_DA_US[:,0], Session_Average_DA_US[:,1])


#testing for normality
W,p_shapiro5 = stats.shapiro(Early_Average_DA_CS[:,0]-Late_Average_DA_CS[:,0])
#we use dependent t-test 
t_shortITI_CS, p_shortITI_CS = stats.ttest_rel(Early_Average_DA_CS[:,0], Late_Average_DA_CS[:,0])

#testing for normality
W,p_shapiro6 = stats.shapiro(Early_Average_DA_US[:,0]-Late_Average_DA_US[:,0])
#we use dependent t-test
t_shortITI_US, p_shortITI_US = stats.ttest_rel(Early_Average_DA_US[:,0], Late_Average_DA_US[:,0])


#testing for normality
W,p_shapiro7 = stats.shapiro(Early_Average_DA_CS[:,1]-Late_Average_DA_CS[:,1])
#we use dependent t-test
t_longITI_CS, p_longITI_CS = stats.ttest_rel(Early_Average_DA_CS[:,1], Late_Average_DA_CS[:,1])

# testing for normality
W,p_shapiro8 = stats.shapiro(Early_Average_DA_US[:,1]-Late_Average_DA_US[:,1]) #p=0.0452
#we use dependent t-test and a wilcoxon test because of slight violation of normality
t_longITI_US, p_longITI_US = stats.ttest_rel(Early_Average_DA_US[:,1], Late_Average_DA_US[:,1])
W_slongITI_US, p_longITI_US_wilcoxon = stats.wilcoxon(Early_Average_DA_US[:,1], Late_Average_DA_US[:,1])