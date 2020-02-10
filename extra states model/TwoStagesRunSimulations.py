# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:13:51 2018

@author: francois
"""

from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, \
MB_component, NextState, LastState
from TwoStagesTaskVariables import n_states, n_actions, n_features, s0, s1, s2, \
s3, s7, goL, goM, wait, E, M, SxA_S, SxA_F, SxA_R
import numpy as np
import matplotlib.pyplot as plt


# experimental setup: 16 sessions of 25 trials which were combined into blocks of 50 trials for averaging

n_trials = 25
n_blocks = 10
n_rats = 20
Distribution_name = 'Beta distribution'
flupenthixol = 0
injection = ''

alpha = 0.03
beta = 0.15
gamma = 0.8
u_iti = 0.01

if Distribution_name == 'Beta distribution':
    omegas = np.random.beta(8,4,n_rats)
elif Distribution_name == 'Uniform distribution':
    omegas = np.random.uniform(0,1, n_rats)
plt.figure(1)
plt.hist(omegas)
plt.xlim(0,1)
plt.savefig(Distribution_name + '/Distribution of omegas.eps')

# initializing transition, reward, value and Q functions


goL_counter1 = np.zeros((n_rats, n_blocks))
goM_counter1 = np.zeros((n_rats, n_blocks))
goL_counter2 = np.zeros((n_rats, n_blocks))
goM_counter2 = np.zeros((n_rats, n_blocks))

Distribution = np.zeros((n_trials * n_blocks * 2, 2, n_rats))
Pvalues = np.zeros((n_trials * n_blocks * 2, 2, n_rats))
choices = np.zeros((n_trials * n_blocks * 2, n_rats))

for rat in range(n_rats):
    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    V = np.zeros((n_features))
    Q = np.zeros((n_states, n_actions))
    
    trial_index = 0
    omega = omegas[rat]
    
    for block in range(n_blocks):


        for t in range(n_trials):
            state = s0
            
            while state != LastState(n_states):
                actions = PossibleActions(SxA_S, state)
                if len(actions)>1:                    
                    IntegrateResults = ActionDistribution(SxA_S, SxA_F, state, omega, alpha, gamma, Q, T, R, V, beta, flupenthixol, injection)
                    Distribution[trial_index, :, rat] = IntegrateResults['Action probabilities']
                    action = IntegrateResults['Choice']
                    Pvalues[trial_index, :, rat] = IntegrateResults['Pvalues']
                    choices[trial_index, rat] = action
                    trial_index += 1
                    if action == goL:
                        if state == s1:
                            goL_counter1[rat, block] += 1
                        else:
                            goL_counter2[rat, block] += 1     
                    elif action == goM:
                        if state == s1:
                            goM_counter1[rat, block] += 1
                        else:
                            goM_counter2[rat, block] += 1
                    elif action == wait:
                        if state == s2:
                            goL_counter2[rat, block] += 1
                        elif state == s3:
                            goM_counter2[rat, block] += 1
                else :
                    action = actions
                FMFResults = FMF_component(SxA_R, SxA_F, SxA_S, state, action, V, alpha, gamma, flupenthixol)
                V = FMFResults['Value']
                ModelBasedResults = MB_component(SxA_R, SxA_F, SxA_S, state, action, Q, T, R, alpha, gamma)
                Q = ModelBasedResults['Qvalues']
                T = ModelBasedResults['Transition function']
                R = ModelBasedResults['MB reward function']
                state = NextState(SxA_S, state, action)
            else:
                # intertrial devaluation of magazine
                action = PossibleActions(SxA_S, state)
                FMFResults = FMF_component(SxA_R, SxA_F, SxA_S, state, action, V, alpha, gamma, flupenthixol)
                V = FMFResults['Value']
                ModelBasedResults = MB_component(SxA_R, SxA_F, SxA_S, state, action, Q, T, R, alpha, gamma)
                Q = ModelBasedResults['Qvalues']
                T = ModelBasedResults['Transition function']
                R = ModelBasedResults['MB reward function']
                
                V[E] = (1 - u_iti) * V[E]
                V[M] = (1 - u_iti) * V[M]

                

filename = Distribution_name + '/Simulations with intertrial ' + str(u_iti)
np.savez(filename, Distribution=Distribution, choices=choices, 
         goL_counter1=goL_counter1, goM_counter1=goM_counter1,
         goL_counter2=goL_counter2, goM_counter2=goM_counter2)