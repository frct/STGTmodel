#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:51:51 2018

simulations of flupenthixol inhibition as in fig 8 of "Modeling individual 
differences in the form of ..." Lesaint et al 2014

@author: francois
"""

from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, MB_component, NextState
from TaskVariables import n_states, n_actions, n_features, s0, s1, s2, s3, s4, s7, goE, \
goL, goM, E, L, M, SxA_S
import numpy as np


# experimental setup: 7 sessions of 50 trials under flupenthixol, 1 final session without flupenthixol

n_trials = 50
n_blocks = 8
test_block = n_blocks - 1
n_rats = 14

# variable options

phenotype = "IG"
iti_scale = 1
Reset_Flupenthixol = 0.8
injection = 'local' #local or systemic otherwise, in the first case only the FMF module is affected
magazine_present = True
lever_present = False


# parameters of the different phenotype as found in Lesaint's articles

if phenotype == "ST":
    # ST properties Lesaint 2014
    alpha = 0.031
    gamma = 0.996
    omega = 0.499
    beta = 0.239
    u_iti = 0.27 # originally 0.027 but I suspect a typo
    # ST properties Lesaint 2015
#    alpha = 0.027
#    gamma = 0.946
#    omega = 0.501
#    beta = 0.243
#    u_iti = 0.845
elif phenotype == "GT":
    #GT properties Lesaint 2014
    alpha = 0.895
    gamma = 0.727
    omega = 0.048
    beta = 0.084
    u_iti = 0.140
    #GT properties Lesaint 2015
#    alpha = 0.033
#    gamma = 0.483
#    omega = 0.081
#    beta = 0.063
#    u_iti = 0.893
elif phenotype == "IG":
    # intermediate group properties Lesaint 2014
    alpha = 0.217
    gamma = 0.999
    omega = 0.276
    beta = 0.142
    u_iti = 0.228
    # intermediate group properties Lesaint 2015
#    alpha = 0.885
#    gamma = 0.989
#    omega = 0.095
#    beta = 0.241
#    u_iti = 0.840


# initializing transition, reward, value and Q functions


goL_counter = np.zeros((n_rats, n_blocks))
goM_counter = np.zeros((n_rats, n_blocks))
exp_counter = np.zeros((n_rats, n_blocks))

Distribution = np.zeros((n_trials * n_blocks, 3, n_rats))
Pvalues = np.zeros((n_trials * n_blocks, 3, n_rats))
choices = np.zeros((n_trials * n_blocks, n_rats))
dopamineCS = np.zeros((n_trials, n_blocks, n_rats))
dopamineUS = np.zeros((n_trials, n_blocks, n_rats))

for rat in range(n_rats):
    
    flupenthixol = Reset_Flupenthixol
    
    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    V = np.zeros((n_features))
    Q = np.zeros((n_states, n_actions))

    if phenotype == "ST":
        # ST initializations
        Q[s1, goL] = 0.844
        Q[s1, goM] = 0.538
        Q[s1, goE] = 0.999
        # ST initializations in Lesaint 2015
#        Q[s1, goL] = 0.263
#        Q[s1, goM] = 0.344
#        Q[s1, goE] = 0.272
    elif phenotype == "GT":
        #GT initializations
        Q[s1, goL] = 1.0
        Q[s1, goM] = 0.023
        Q[s1, goE] = 0.316
        #GT initializations in Lesaint 2015
#        Q[s1, goL] = 0.936
#        Q[s1, goM] = 0.099
#        Q[s1, goE] = 0.022
    elif phenotype == "IG":
        # IG initializations
        Q[s1, goL] = 0.526
        Q[s1, goM] = 0.587
        Q[s1, goE] = 0.888
        # IG initializations in Lesaint 2015
#        Q[s1, goL] = 0.059
#        Q[s1, goM] = 0.732
#        Q[s1, goE] = 0.142
    
    for b in range(n_blocks):

        if b == test_block:
            flupenthixol = 0
            
        
        
        for t in range(n_trials):
            state = s0
            trial_index = b * n_trials + t
            passed_by_s4 = False
            while state != s7:

                if state == s1:
                    IntegrateResults = ActionDistribution(state, omega, alpha, gamma, Q, T, R, V, beta, flupenthixol, injection)
                    Distribution[trial_index, :, rat] = IntegrateResults['Action probabilities']
                    action = IntegrateResults['Choice']
                    Pvalues[trial_index, :, rat] = IntegrateResults['Pvalues']
                    choices[trial_index, rat] = action
                    if action == goL:
                        goL_counter[rat, b] += 1
                    elif action == goM:
                        goM_counter[rat, b] += 1
                    elif action == goE:
                        exp_counter[rat, b] += 1
                else :
                    action = PossibleActions(SxA_S, state)
                FMFResults = FMF_component(state, action, V, alpha, gamma, flupenthixol)
                V = FMFResults['Value']
                if state == s0:
                    dopamineCS[t, b, rat] = FMFResults['DA']
                elif state == s2 or state == s3 or state == s4:# calculate US RPE based on state preceding appearance of food
                    dopamineUS[t, b, rat] = FMFResults['DA']
                ModelBasedResults = MB_component(state, action, Q, T, R, alpha, gamma)
                Q = ModelBasedResults['Qvalues']
                T = ModelBasedResults['Transition function']
                R = ModelBasedResults['MB reward function']
                state = NextState(state, action)
                
            else:
                # intertrial devaluation of magazine
                action = PossibleActions(SxA_S, state)
                FMFResults = FMF_component(state, action, V, alpha, gamma, flupenthixol)
                V = FMFResults['Value']
#                if passed_by_s4:
#                    # either calculate RPE of US based on state immediately preceding eating food
                #dopamineUS[t, b, rat] = FMFResults['DA']
                ModelBasedResults = MB_component(state, action, Q, T, R, alpha, gamma)
                Q = ModelBasedResults['Qvalues']
                T = ModelBasedResults['Transition function']
                R = ModelBasedResults['MB reward function']
                
                V[E] = (1 - iti_scale * u_iti) * V[E]
                # if magazine not removed during inter-trial
                if magazine_present:
                    V[M] = (1 - iti_scale * u_iti) * V[M]
                if lever_present :
                    V[L] = (1 - iti_scale * u_iti) * V[L]
                

#filename = 'flupenthixol inhibition/' + phenotype + ' flupenthixol inhibition = ' + str(Reset_Flupenthixol) + '.npz'
filename = 'replication Lesaint 2014/' + phenotype + ' flupenthixol inhibition = ' + str(Reset_Flupenthixol) + '.npz'
np.savez(filename, Distribution=Distribution, choices=choices, 
         goL_counter=goL_counter, goM_counter=goM_counter, exp_counter=exp_counter,
         dopamineCS=dopamineCS, dopamineUS=dopamineUS)