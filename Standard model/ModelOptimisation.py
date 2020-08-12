# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:56:01 2020

@author: franc
"""

import numpy as np
from scipy import optimize
from TaskVariables import state0, state1, state7, N, M, L, F, n_features, n_states, \
n_actions, goE, goM
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
            likelihood = Distribution[action-1]
        else :
            action = PossibleActions(state)
        FMFResults = FMF_component(state, action, Estimates.V, Parameters)
        Estimates.V = FMFResults['Value']
        ModelBasedResults = MB_component(state, action,Estimates, Parameters)
        
        Estimates.Q = ModelBasedResults.Q
        Estimates.T = ModelBasedResults.T
                
        state = NextState(state, action)
    else:
        # intertrial devaluation of magazine
        action = PossibleActions(state)
        
        FMFResults = FMF_component(state, action, Estimates.V, Parameters)
        Estimates.V = FMFResults['Value']
        ModelBasedResults =  MB_component(state, action, Estimates, Parameters)
        Estimates.Q = ModelBasedResults.Q
        Estimates.T = ModelBasedResults.T
        Estimates.R = ModelBasedResults.R
        #print('values before:' + str(Estimates.V))
        Estimates.V[N] = (1 - Parameters.u_iti) * Estimates.V[N]
        Estimates.V[M] = (1 - Parameters.u_iti) * Estimates.V[M]
        #print('values after:' + str(Estimates.V))
    return {'likelihood': likelihood, 'Estimates': Estimates}

    
def LogLikelihood(data, Estimates, Parameters):
    n_trials = len(data)
    TrialLikelihood = np.zeros((n_trials))
    
    for t in range(n_trials):
        action = data[t]
        TrialUpdates = SingleTrial(action, Parameters, Estimates)
        TrialLikelihood[t] = TrialUpdates['likelihood']
        Estimates = TrialUpdates['Estimates']
        
    LogLikelihood = sum(np.log(TrialLikelihood))
    
    return LogLikelihood

def OptimizeOmega(omega):
    Parameters = structure()
    Parameters.alpha = 0.2
    Parameters.beta = 0.09
    Parameters.gamma = 0.8
    Parameters.u_iti = 0.2
    Parameters.omega = omega

    InitialEstimates = structure()
    InitialEstimates.V = np.zeros((n_features))
    InitialEstimates.V[F] = 1
    
    InitialEstimates.Q = np.zeros((n_states, n_actions))
    InitialEstimates.Q[state1,goE] = 0.5
    InitialEstimates.Q[state1, goM] = 0.5
    InitialEstimates.T = np.zeros((n_states, n_actions, n_states))
    InitialEstimates.R = np.zeros((n_states, n_actions))

    negative_LogLikelihood = -1 * LogLikelihood(rat_data, InitialEstimates, Parameters)
    
    return negative_LogLikelihood

def OptimizeParameters(x):
    Parameters = structure()
    Parameters.alpha = x[0]
    Parameters.beta = x[1]
    Parameters.gamma = x[2]
    Parameters.u_iti = x[3]
    Parameters.omega = x[4]
    
    InitialEstimates = structure()
    InitialEstimates.V = np.zeros((n_features))
    InitialEstimates.V[F] = 1
    
    InitialEstimates.Q = np.zeros((n_states, n_actions))
    InitialEstimates.Q[state1,goE] = 0.5
    InitialEstimates.Q[state1, goM] = 0.5
    InitialEstimates.T = np.zeros((n_states, n_actions, n_states))
    InitialEstimates.R = np.zeros((n_states, n_actions))

    negative_LogLikelihood = -1 * LogLikelihood(rat_data, InitialEstimates, Parameters)
    
    return negative_LogLikelihood

import scipy.io
mat = scipy.io.loadmat('Compiled_STGT_model_data.mat')
data = mat['Compiled_STGT_mdl_data']

rats = np.unique(data[:,0])
n_rats = len(rats)

omegas = np.zeros((n_rats))
negLL = np.zeros((n_rats))
avgLikelihood = np.zeros((n_rats))
params = np.zeros((n_rats, 5))

for index, rat in enumerate(rats):
    b = data[:,0] == rat
    rat_data = data[b,3]
    n_trials = len(rat_data)
    
    '''x0 = [0.1]
    bnds = ((0,1),(0,1))
    #result = optimize.minimize(OptimizeOmega, x0)
    result = optimize.minimize_scalar(OptimizeOmega, bounds=(0,1), method='bounded')
    negLL[index] = result.fun
    omegas[index] = result.x
    avgLikelihood[index] = np.exp(-1*negLL[index] / n_trials)'''
    
    x0 = np.array([0.2, 0.09, 0.8,0.2,0.5])
    bnds = ((0,1), (0.001, 1), (0,1), (0,1), (0,1))
    result = optimize.minimize(OptimizeParameters, x0, bounds=bnds)
    negLL[index] = result.fun
    avgLikelihood[index] = np.exp(-1*negLL[index] / n_trials)
    params[index,:] = result.x