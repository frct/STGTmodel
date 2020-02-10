#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:45:52 2018

runs simulations of a model with decaying MB input and increasing combination 
of reward and food cup value in s1

@author: francois
"""

from ModelDefinition import Feature, ActionDistribution, PossibleActions, FMF_component, MB_component, NextState
from TaskVariables import n_states, n_actions, n_features, s0, s1, s2, s3, s4, s5, \
eng, goL, goM, O, L, M, R
import numpy as np


# experimental setup: 16 sessions of 25 trials which were combined into blocks of 50 trials for averaging

n_trials = 50 #50
n_blocks = 8
n_rats = 14

# variable options

flupenthixol = 0
magazine_present = True
lever_present = False
injection = ''


# parameters of the different phenotypes: using shared parameters and playing with omega only

alpha = 0.20
gamma = 0.8
beta = 0.09
u_iti = 1

omegas = np.linspace(0, 1, 8)



# initializing transition, reward, value and Q functions


goL_counter1 = np.zeros((n_rats, n_blocks))
goL_counter2 = np.zeros((n_rats, n_blocks))
goM_counter1 = np.zeros((n_rats, n_blocks))
goM_counter2 = np.zeros((n_rats, n_blocks))

Distribution = np.zeros((n_trials * n_blocks, 2, n_rats))
Pvalues = np.zeros((n_trials * n_blocks, 2, n_rats))
choices1 = np.zeros((n_trials * n_blocks, n_rats))
choices2 = float('nan') * np.ones((n_trials * n_blocks, n_rats))
dopamineCS = np.zeros((n_trials, n_blocks, n_rats))
dopamineUS = np.zeros((n_trials, n_blocks, n_rats))


for rat in range(n_rats):
    
    
    Transition_function = np.zeros((n_states, n_actions, n_states))
    Reward_function = np.zeros((n_states, n_actions))
    Q = np.zeros((n_states, n_actions))
    V = np.zeros((n_features))
    
    for b in range(n_blocks):
        omega = omegas[b]
#        omega = 0
        for t in range(n_trials):
            # hypothesis that the value of food does not need to be learnt
            # and is clamped to 1 due to potential devaluation when transitioning
            # from state 4 to state 5
            V[R] = 1 
            
            state = s0
            trial_index = b * n_trials + t
            
            while state != s5:
#                print('V in state ' +str(state) + ':' + str(V))
                
                if state == s1:
#                    transientV = V.copy()
#                    transientV[M] = (1 - omega) * V[M] + omega * V[R] # we use omega as a weight so that only GTers make the mistake of confusing V[M] with V[R] at this point
                    IntegrateResults = ActionDistribution(state, V[M], alpha, gamma, Q, Transition_function, Reward_function, V, beta, flupenthixol, injection)
                    Distribution[trial_index, :, rat] = IntegrateResults['Action probabilities']
                    action = IntegrateResults['Choice']
                    choices1[trial_index, rat] = action
                    if choices1[trial_index, rat] == goL:
                        goL_counter1[rat, b] += 1
                    elif choices1[trial_index, rat] == goM:
                        goM_counter1[rat, b] += 1
                elif state == s3:
                    #print('V:' + str(V))
                    IntegrateResults = ActionDistribution(state, omega, alpha, gamma, Q, Transition_function, Reward_function, V, beta, flupenthixol, injection)
                    Distribution[trial_index, :, rat] = IntegrateResults['Action probabilities']
                    action = IntegrateResults['Choice']
                    choices2[trial_index, rat] = action
                    if choices2[trial_index, rat] == goL:
                        goL_counter2[rat, b] += 1
                    elif choices2[trial_index, rat] == eng:
                        goM_counter2[rat, b] += 1
                else :
                    if state == s2:
                        goL_counter2[rat, b] += 1
                    action = PossibleActions(state)
                    
                # learning from results 
                FMFResults = FMF_component(state, action, V, alpha, gamma, flupenthixol)
                V = FMFResults['Value']
                ModelBasedResults = MB_component(state, action, Q, Transition_function, Reward_function, alpha, gamma)
                Q = ModelBasedResults['Qvalues']
                Transition_function = ModelBasedResults['Transition function']
                Reward_function = ModelBasedResults['MB reward function']
                
#                if state != s3:
#                    V = FMFResults['Value']
                if state == s0:
                    dopamineCS[t, b, rat] = FMFResults['DA']
                elif state == s2 or (state == s3 and action == eng):#either calculate US RPE based on state preceding appearance of food
                    dopamineUS[t, b, rat] = FMFResults['DA']

 
                state = NextState(state, action)
                
            else:
#                print('V in state ' +str(state) + ':' + str(V))
                # intertrial devaluation of magazine
                action = PossibleActions(state)
                FMFResults = FMF_component(state, action, V, alpha, gamma, flupenthixol)
                V = FMFResults['Value']
                ModelBasedResults = MB_component(state, action, Q, Transition_function, Reward_function, alpha, gamma)
                Q = ModelBasedResults['Qvalues']
                Transition_function = ModelBasedResults['Transition function']
                Reward_function = ModelBasedResults['MB reward function']
                V[O] = (1 -  u_iti) * V[O]
                # if magazine not removed during inter-trial
                if magazine_present:
                    V[M] = (1 - u_iti) * V[M]
                

filename = 'Decaying MB Simulations.npz'

np.savez(filename, Distribution=Distribution, choices1=choices1, choices2=choices2,
         goL_counter1=goL_counter1, goL_counter2=goL_counter2,
         goM_counter1=goM_counter1, goM_counter2=goM_counter2,
         dopamineCS=dopamineCS, dopamineUS=dopamineUS)