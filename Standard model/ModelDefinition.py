# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:51:38 2018

@author: cinotti
"""

import numpy as np
#from TaskVariables import SxA_F, SxA_S, SxA_R, n_states

def LastState(n_states):
    return n_states - 1

def Feature(SxA_F, state,action):
    # returns feature for a given state and action
    return int(SxA_F[state, action])

def PossibleFeatures(SxA_F, state):
    # returns the possible features corresponding to a given state
    f = SxA_F[state, :]
    f = f[~np.isnan(f)]
    return f.astype(int)

def NextState(SxA_S, state, action):
    # returns next state for a given action * state
    return int(SxA_S[state, action])

def PossibleActions(SxA_S, state):
    # returns the possible actions in a given state
    f = SxA_S[state, :]
    actions = np.argwhere(~np.isnan(f))
    return actions

def FMF_component(SxA_R, SxA_F, SxA_S, state, action, V, alpha, gamma, flupenthixol):
    n_states = np.size(SxA_R,0)
    c = Feature(SxA_F, state, action)
    r = SxA_R[state, action]
    if state != LastState(n_states):
        next_features = PossibleFeatures(SxA_F, NextState(SxA_S, state, action))
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
#    
#    if state == LastState(n_states):
#        delta = r - V[2]
    
    return {'Value' : V, 'DA' : delta}


def MB_component(SxA_R, SxA_F, SxA_S, state, action, Q, T, R, alpha, gamma):
    n_states = np.size(SxA_R,0)
    r = SxA_R[state, action]
    R[state, action] = R[state, action] + alpha * (r - R[state, action])
    if state != LastState(n_states): # task is episodic, transition from last state to state 0 must be treated differently
        next_state = NextState(SxA_S, state, action)
        T[state, action, :] = (1 - alpha) * T[state, action, :]
        T[state, action, next_state] += alpha
        maxQ = np.amax(Q, axis = 1)
        Q[state, action] = R[state, action] + gamma * np.dot(T[state, action, :], maxQ)
    else :
        Q[state, action] = R[state, action]
    
    return {'Transition function': T, 'MB reward function' : R, 'Qvalues': Q}


def ActionDistribution(SxA_S, SxA_F, state, omega, alpha, gamma, Q, T, R, V, beta, flupenthixol, injection):
    actions = PossibleActions(SxA_S, state)
    n_possibleactions = len(actions)
    P = np.zeros(n_possibleactions)
    for index, action in enumerate(actions):
        c = Feature(SxA_F, state, action)
        Advantage = Q[state, action] - max(Q[state, :])
        P[index] = (1 - omega) * Advantage + omega * V[c]
        
    distribution = np.zeros(n_possibleactions)
    if injection == 'systemic':
        beta = beta / (1 - flupenthixol)
    for i in range(n_possibleactions):
        distribution[i] = np.exp(P[i] / beta) / sum(np.exp(P[:] / beta))
    j = np.random.choice(n_possibleactions, 1, p=distribution)
    choice = actions[j]
    return {'Action probabilities':distribution, 'Choice': choice, 'Pvalues': P}