# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:56:01 2020

@author: franc
"""

import numpy as np
from TaskVariables import state0, state1, state7, E, M, F, n_features, n_states, \
n_actions
from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, \
MB_component, NextState

class structure:
    pass


def SingleTrial(State1_Action, Parameters, Estimates):
    state = state0
    # Estimates.V[F] = 1 # hypothesis that the value of food does not need to be learnt and is clamped...
    while state != state7:
        if state == state1:
            action = State1_Action
            IntegrateResults = ActionDistribution(state, Parameters, Estimates)
            Distribution = IntegrateResults['Action probabilities']
            likelihood = Distribution[action]
        else :
            action = PossibleActions(state)
        print('State:' + str(state) + ', Action:' + str(action))
        FMFResults = FMF_component(state, action, Estimates.V, Parameters)
        Estimates.V = FMFResults['Value']
        ModelBasedResults = MB_component(state, action,Estimates, Parameters)
        Estimates.Q = ModelBasedResults['Qvalues']
        Estimates.T = ModelBasedResults['Transition function']
                
        state = NextState(state, action)
    else:
        # intertrial devaluation of magazine
        action = PossibleActions(state)
        FMFResults = FMF_component(state, action, Estimates.V, Parameters)
        Estimates.V = FMFResults['Value']
        ModelBasedResults =  MB_component(state, action, Estimates, Parameters)
        Estimates.Q = ModelBasedResults['Qvalues']
        Estimates.T = ModelBasedResults['Transition function']
        Estimates.R = ModelBasedResults['MB reward function']  
        Estimates.V[E] = (1 - Parameters.u_iti) * Estimates.V[E]
        Estimates.V[M] = (1 - Parameters.u_iti) * Estimates.V[M]
    return {'likelihood': likelihood, 'Estimates': Estimates}
        

def LogLikelihood(data, omega):
    n_trials = len(data)
    TrialLikelihood = np.zeros((n_trials))
    
    Estimates = structure()
    Estimates.V = np.zeros((n_features))
    Estimates.Q = np.zeros((n_states, n_actions))
    Estimates.T = np.zeros((n_states, n_actions, n_states))
    Estimates.R = np.zeros((n_states, n_actions))
    
    Parameters = structure()
    Parameters.alpha = 0.2
    Parameters.beta = 0.09
    Parameters.gamma = 0.8
    Parameters.u_iti = 0.2
    
    for t in range(n_trials):
        action = data[t, 3]
        TrialUpdates = SingleTrial(action, Parameters, Estimates)
        TrialLikelihood[t] = TrialUpdates['likelihood']
        
    LogLikelihood = sum(np.log(TrialLikelihood))
    
    return LogLikelihood