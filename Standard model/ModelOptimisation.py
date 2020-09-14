# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:56:01 2020

@author: franc
"""

import numpy as np
from scipy import optimize
from TaskVariables import state0, state1, state5, N, M, F, n_features, n_states, \
n_actions, goM
from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, \
MB_component, NextState

class structure:
    pass


def SingleTrial(State1_Action, Parameters, Estimates):
    state = state0
    while state != state5:
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
        Estimates.V[N] = (1 - Parameters.u_iti) * Estimates.V[N]
        Estimates.V[M] = (1 - Parameters.u_iti) * Estimates.V[M]
        
        ModelBasedResults =  MB_component(state, action, Estimates, Parameters)
        Estimates.Q = ModelBasedResults.Q
        Estimates.T = ModelBasedResults.T
        Estimates.R = ModelBasedResults.R
        
    return {'likelihood': likelihood, 'Estimates': Estimates}

    
def LogLikelihood(data, Estimates, Parameters):
    n_trials = len(data)
    TrialLikelihood = np.zeros((n_trials))
    
    for t in range(n_trials):
        action = int(data[t])
        TrialUpdates = SingleTrial(action, Parameters, Estimates)
        TrialLikelihood[t] = TrialUpdates['likelihood']
        Estimates = TrialUpdates['Estimates']
        
    LogLikelihood = sum(np.log(TrialLikelihood))
    
    return LogLikelihood

def OptimizeOmega(omega):
    Parameters = structure()
    Parameters.alpha_MF = 0.2
    Parameters.alpha_MB = 0.2
    Parameters.beta = 4
    Parameters.gamma = 1
    Parameters.u_iti = 0.2
    Parameters.omega = omega

    InitialEstimates = structure()
    InitialEstimates.V = np.zeros((n_features))
    InitialEstimates.V[F] = 1
    
    InitialEstimates.Q = np.zeros((n_states, n_actions))
    #InitialEstimates.Q[state1, goM] = 0.5
    InitialEstimates.T = np.zeros((n_states, n_actions, n_states))
    InitialEstimates.R = np.zeros((n_states, n_actions))

    negative_LogLikelihood = -1 * LogLikelihood(rat_data, InitialEstimates, Parameters)
    
    return negative_LogLikelihood

def OptimizeParameters(x):
    Parameters = structure()
    # Parameters.alpha = x[0]
    # Parameters.beta = x[1]
    # Parameters.gamma = x[2]
    # Parameters.u_iti = x[3]
    # Parameters.omega = x[4]
    
    Parameters.alpha_MB = x[0]
    Parameters.alpha_MF = x[0]
    Parameters.beta = x[1]
    Parameters.gamma = 1
    Parameters.u_iti = x[2]
    Parameters.omega = x[3]
    
    # Parameters.alpha = x[0]
    # Parameters.w_FMF = x[1]
    # Parameters.w_MB = x[2]
    # Parameters.u_iti = x[3]
    # Parameters.gamma = 1
    
    # Parameters.alpha_MB = x[0]
    # Parameters.alpha_MF = x[1]
    # Parameters.beta = x[2]
    # Parameters.u_iti = x[3]
    # Parameters.omega = x[4]
    # Parameters.gamma = 1
    
    InitialEstimates = structure()
    InitialEstimates.V = np.zeros((n_features))
    InitialEstimates.V[F] = 1
    
    InitialEstimates.Q = np.zeros((n_states, n_actions))
    #InitialEstimates.Q[state1, goM] = 0.5
    InitialEstimates.T = np.zeros((n_states, n_actions, n_states))
    InitialEstimates.R = np.zeros((n_states, n_actions))

    negative_LogLikelihood = -1 * LogLikelihood(rat_data, InitialEstimates, Parameters)
    
    return negative_LogLikelihood

def InitialisationIndices(n_parameters):
    nx = 3**n_parameters

    I = np.zeros((nx, n_parameters))

    for idy in range(n_parameters): # sweeping through columns
        idx = 0
        for n_reps in range(3 ** idy): # each column is made of repeated sequences of groups of 1's 2's 3's. The first colums containts only one repetition, the next 3, the next 9, etc...
            for a in range(3):
                for b in range(3**(n_parameters - idy-1)): # size of this repetition
                    I[idx, idy] = a
                    idx = idx + 1
                    
    return I.astype(int)



data = np.load('Binary classification of data.npy')

rats = np.unique(data[:,0])
n_rats = len(rats)



#p = np.array([[0.2, 0.5, 0.8], [1, 2, 5], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]) # with gamma
p = np.array([[0.05, 0.2, 0.5], [1, 2, 5], [0.05, 0.2, 0.5], [0.2, 0.5, 0.8]]) # without gamma 
#p = np.array([[0.2, 0.5, 0.8], [1, 2, 5], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]) # without gamma but with two weighting parameters
#p = np.array([[0.2, 0.5, 0.8], [1, 2, 5], [1, 2, 5], [0.2, 0.5, 0.8]]) # without gamma and beta but with two weighting parameters
#p = np.array([[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [1, 2, 5], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]) # 2 learning rates, beta, u_iti and omega
#p = np.array([0.2, 0.5, 0.8])

#n_parameters = 1
n_parameters = p.shape[0]

I = InitialisationIndices(n_parameters)

for index, rat in enumerate(rats):
    rat = int(rat)
    b = data[:,0] == rat
    rat_data = data[b,3]
    n_trials = len(rat_data)
    
    negLL = np.zeros((3**n_parameters))
    #avgLikelihood = np.zeros((n_rats))
    params = np.zeros((3**n_parameters, n_parameters))
    
    x0 = [0.1]
    #result = optimize.minimize(OptimizeOmega, x0)
    
    
    bnds = ((0,1), (0.001, 100), (0,1), (0,1))
    # bnds = ((0,1), (0,1), (0, 100), (0,1), (0,1))
    
    for i in range(3**n_parameters):
        x0 = np.zeros((n_parameters))
        # x0 = p[i]
        for param in range(n_parameters):
            x0[param] = p[param, I[i, param]]
        
        # result = optimize.minimize_scalar(OptimizeOmega, bounds=(0,1), method='bounded')
        # negLL[i] = result.fun
        # params[i] = result.x
        #avgLikelihood[i] = np.exp(-1*negLL[i] / n_trials)
        
        result = optimize.minimize(OptimizeParameters, x0, bounds=bnds)
        negLL[i] = result.fun
        #avgLikelihood[i] = np.exp(-1*negLL[index] / n_trials)
        params[i,:] = result.x
    np.save('Rat' + str(rat) + ' parameters', params )
    np.save('Rat' + str(rat) + ' negLL', negLL )