# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:56:01 2020

@author: franc
"""

import numpy as np
from scipy import optimize
from TaskVariables import state0, state1, state5, N, M, F, n_features, n_states, \
n_actions, goM, Estimates
from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, \
MB_component, NextState, Parameters

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
        
        ModelBasedResults = MB_component(state, action, Estimates, Parameters)
        Estimates.Q = ModelBasedResults.Q
        Estimates.T = ModelBasedResults.T
                
        state = NextState(state, action)
    else:
        # intertrial devaluation of magazine
        action = PossibleActions(state)
        
        FMFResults = FMF_component(state, action, Estimates.V, Parameters)
        Estimates.V = FMFResults['Value']
        Estimates.V[N] = (1 - Parameters.u) * Estimates.V[N]
        Estimates.V[M] = (1 - Parameters.u) * Estimates.V[M]
        
        ModelBasedResults =  MB_component(state, action, Estimates, Parameters)
        Estimates.Q = ModelBasedResults.Q
        Estimates.T = ModelBasedResults.T
        Estimates.R = ModelBasedResults.R
        
    return {'likelihood': likelihood, 'Estimates': Estimates}

    
def OmegaOnlyLogLikelihood(omega, data, Estimates):
    parameters = Parameters(omega = omega)
    n_trials = len(data)
    TrialLikelihood = np.zeros((n_trials))
    for t in range(n_trials):
        action = int(data[t])
        TrialUpdates = SingleTrial(action, parameters, Estimates)
        TrialLikelihood[t] = TrialUpdates['likelihood']
        Estimates = TrialUpdates['Estimates']
        
    LogLikelihood = sum(np.log(TrialLikelihood))
    
    return LogLikelihood

def OptimizeOmega(data):
    InitialEstimates = Estimates()  
    res = optimize.minimize_scalar(
            fun=lambda omega, data, InitialEstimates : - OmegaOnlyLogLikelihood(omega, data, InitialEstimates),
            args=(data, InitialEstimates),
            bounds=(0,1),
            method='bounded')
    
    return res

def LogLikelihood(x, data, estimates):
    parameters = Parameters(alpha = x[0], beta = x[1], omega = x[2], u = x[3])
    n_trials = len(data)
    TrialLikelihood = np.zeros((n_trials))
    for t in range(n_trials):
        action = int(data[t])
        TrialUpdates = SingleTrial(action, parameters, estimates)
        TrialLikelihood[t] = TrialUpdates['likelihood']
        estimates = TrialUpdates['Estimates']
        
    LogLikelihood = sum(np.log(TrialLikelihood))
    
    return LogLikelihood

def OptimizeParameters(x, data):
    bnds = ((0,1), (0.001, 100), (0,1), (0,1))
    estimates = Estimates()
    res = optimize.minimize(
        fun=lambda parameters, data, InitialEstimates : - LogLikelihood(parameters, data, estimates),
        args=(data, estimates),
        x0=x,
        bounds = bnds)
    
    return res

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