# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:51:38 2018

@author: cinotti
"""

import numpy as np
from TaskVariables import SxA_F, SxA_S, SxA_R, n_states

def LastState():
    return n_states - 1

def Feature(state,action):
    # returns feature for a given state and action
    return int(SxA_F[state, action])

def PossibleFeatures(state):
    # returns the possible features corresponding to a given state
    f = SxA_F[state, :]
    f = f[~np.isnan(f)]
    return f.astype(int)

def NextState(state, action):
    # returns next state for a given action * state
    return int(SxA_S[state, action])

def PossibleActions(state):
    # returns the possible actions in a given state
    f = SxA_S[state, :]
    actions = np.argwhere(~np.isnan(f))
    return actions

def FMF_component(state, action, V, alpha, gamma, flupenthixol):
    c = Feature(state, action)
    r = SxA_R[state, action]
    if state != LastState():
        next_features = PossibleFeatures(NextState(state, action))
        delta = r + gamma * max(V[next_features]) - V[c]
    else :
        delta = r - V[c]
        
    # inhibition by flupenthixol
    if delta != 0:
        if (delta - flupenthixol) / delta >= 0:
            delta -= flupenthixol
        else :
            delta = 0    
    V[c] = V[c] + alpha * delta
    return {'Value' : V, 'DA' : delta}


def MB_component(state, action, Q, T, R, alpha, gamma):
    r = SxA_R[state, action]
    R[state, action] = R[state, action] + alpha * (r - R[state, action])
    if state != LastState(): # task is episodic, transition from last state to state 0 must be treated differently
        next_state = NextState(state, action)
        T[state, action, :] = (1 - alpha) * T[state, action, :]
        T[state, action, next_state] += alpha
        maxQ = np.amax(Q, axis = 1)
        Q[state, action] = R[state, action] + gamma * np.dot(T[state, action, :], maxQ)
    else :
        Q[state, action] = R[state, action]
    
    return {'Transition function': T, 'MB reward function' : R, 'Qvalues': Q}


#def Combine_FMF_and_MB(state, omega, alpha, gamma, Q, T, R, V):
#    actions = PossibleActions(state)
#    n_possibleactions = len(actions)
#    P = np.zeros(n_possibleactions)
#    for index, action in enumerate(actions):
#        c = Feature(state, action)
#        Advantage = Q[state, action] - max(Q[state, :])
#        P[index] = (1 - omega) * Advantage + omega * V[c]
#        #print(P)
#    return {'Integrated values': P, 'Actions': actions}


def ActionDistribution(state, omega, alpha, gamma, Q, T, R, V, beta, flupenthixol, injection):
    actions = PossibleActions(state)
    n_possibleactions = len(actions)
    P = np.zeros(n_possibleactions)
    for index, action in enumerate(actions):
        c = Feature(state, action)
        Advantage = Q[state, action] - max(Q[state, :])
        P[index] = (1 - omega) * Advantage + omega * V[c]
        #print(P)
        
    distribution = np.zeros(n_possibleactions)
    if injection == 'systemic':
        beta = beta / (1 - flupenthixol)
    for i in range(n_possibleactions):
        distribution[i] = np.exp(P[i] / beta) / sum(np.exp(P[:] / beta))
    j = np.random.choice(n_possibleactions, 1, p=distribution)
    choice = actions[j]
    return {'Action probabilities':distribution, 'Choice': choice, 'Pvalues': P}