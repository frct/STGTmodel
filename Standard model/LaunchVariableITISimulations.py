#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:49:50 2018

launch simulations with variable iti devaluations for extreme MB or FMF 
parameterizations

@author: francois
"""

from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, \
MB_component, NextState
from TaskVariables import n_states, n_actions, n_features, s0, s1, s2, s3, s4, s7, goE, \
goL, goM, E, L, M, F, SxA_S, SxA_F, SxA_R
import numpy as np


# experimental setup: 16 sessions of 25 trials which were combined into blocks of 50 trials for averaging

n_trials = 25
n_blocks = 10
n_rats = 20

# variable options

phenotypes = ['ST', 'GT']
n_phenotypes = len(phenotypes)
flupenthixol = 0

# handtuned parameters of the model with shared parameters as reported in Lesaint et al 2014
    
alpha = 0.03
beta = 0.15
gamma = 0.8
u_itis = np.array([0.01, 0.1])


for phenotype in phenotypes:
    
    if phenotype == "ST":
        omega = 1
        
#        # ST properties Lesaint 2015 obtained by optimizing index score as well as fit
#        alpha = 0.027
#        gamma = 0.946
#        beta = 0.243
#        omega = 1 #0.501
        
    elif phenotype == "GT":
        omega = 0
        
        #GT properties Lesaint 2015
#        alpha = 0.033
#        gamma = 0.483
#        beta = 0.063
#        omega = 0.081
        
    FinalQTable = np.zeros((n_states, n_actions, 3))
    
    for iti, u_iti in enumerate(u_itis):
        
        # initializing transition, reward, value and Q functions
        
        
        goL_counter = np.zeros((n_rats, n_blocks))
        goM_counter = np.zeros((n_rats, n_blocks))
        exp_counter = np.zeros((n_rats, n_blocks))
        
        Distribution = np.zeros((n_trials * n_blocks, 3, n_rats))
        Pvalues = np.zeros((n_trials * n_blocks, 3, n_rats))
        choices = np.zeros((n_trials * n_blocks, n_rats))
        dopamineCS = np.zeros((n_trials, n_blocks, n_rats))
        dopamineUS = np.zeros((n_trials, n_blocks, n_rats))
        state1_FMFValue = np.zeros((n_trials * n_blocks, 3, n_rats)) # this is the value of features used for decision making in state 1
        state1_MBAdvantage = np.zeros((n_trials * n_blocks, 3, n_rats))
        
        for rat in range(n_rats):
            T = np.zeros((n_states, n_actions, n_states))
            R = np.zeros((n_states, n_actions))
            V = np.zeros((n_features))
            
            Q = np.zeros((n_states, n_actions))
            
            for b in range(n_blocks):
        
        
                for t in range(n_trials):
                    state = s0
                    trial_index = b * n_trials + t
                    V[F] = 1 # hypothesis that the value of food does not need to be learnt and is clamped...
                    
                    while state != s7:
        
                        if state == s1:
                            IntegrateResults = ActionDistribution(SxA_S, SxA_F, state, omega, alpha, gamma, Q, T, R, V, beta, flupenthixol, 'False')
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
                        #print('Action:' + str(action))
                        FMFResults = FMF_component(SxA_R, SxA_F, SxA_S, state, action, V, alpha, gamma, flupenthixol)
                        V = FMFResults['Value']
                        ModelBasedResults = MB_component(SxA_R, SxA_F, SxA_S, state, action, Q, T, R, alpha, gamma)
                        Q = ModelBasedResults['Qvalues']
                        T = ModelBasedResults['Transition function']
                        R = ModelBasedResults['MB reward function']
                        
                        
                        if state == s0:
                            dopamineCS[t, b, rat] = FMFResults['DA']
                            state1_FMFValue[trial_index, :, rat] = V[np.array([E,L,M])]
                            state1_MBAdvantage[trial_index, :, rat] = Q[s1, [goE, goL, goM]] - max(Q[1, [goE, goL, goM]])
                        elif state == s2 or state == s3 or state == s4:#either calculate US RPE based on state preceding appearance of food
                            dopamineUS[t, b, rat] = FMFResults['DA']
                            
                        state = NextState(SxA_S, state, action)
                    else:
                        # intertrial devaluation of magazine
                        action = PossibleActions(SxA_S, state)
                        FMFResults = FMF_component(SxA_R, SxA_F, SxA_S, state, action, V, alpha, gamma, flupenthixol)
                        V = FMFResults['Value']
        #                if passed_by_s4:
        #                    # either calculate RPE of US based on state immediately preceding eating food
                        #dopamineUS[t, b, rat] = FMFResults['DA']
                        ModelBasedResults =  MB_component(SxA_R, SxA_F, SxA_S, state, action, Q, T, R, alpha, gamma)
                        Q = ModelBasedResults['Qvalues']
                        T = ModelBasedResults['Transition function']
                        R = ModelBasedResults['MB reward function']
                        
                        V[E] = (1 - u_iti) * V[E]
                        V[M] = (1 - u_iti) * V[M]
                        
            FinalQTable[:,:,iti] = Q
                
        filename = 'Variable ITIs/' + phenotype + 'Simulations with intertrial ' + str(iti) + '.npz'
        
        np.savez(filename, Distribution=Distribution, choices=choices, 
                 goL_counter=goL_counter, goM_counter=goM_counter, exp_counter=exp_counter,
                 dopamineCS=dopamineCS, dopamineUS=dopamineUS, state1_FMFValue=state1_FMFValue, 
                 state1_MBAdvantage=state1_MBAdvantage, Pvalues=Pvalues )