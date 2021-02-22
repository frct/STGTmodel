# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:25:05 2021

Launch simulations of the model

@author: franc
"""

# %% prelude

import numpy as np
from ModelDefinition import ActionDistribution, PossibleActions, FMF_component, \
MB_component, NextState
from TaskVariables import F, state1, goM, state0, state5, N, M

data = np.load('Binary classification of data.npy')

rats = np.unique(data[:,0])
n_rats = len(rats)

class Parameters(object):
    def __init__(self, alpha = 0.2, beta = 4., gamma = 1., u = 0.2, omega=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.u = u
        self.omega = omega
        
    def __repr__(self):
        return "(alpha = %s, beta = %s, gamma = %s, u = %s, omega = %s)" %(self.alpha, self.beta, self.gamma, self.u, self.omega)

class Estimates(object):
    def __init__(self, n_features=4, n_states=6, n_actions=5):
        self.V = np.zeros((n_features))
        self.V[F] = 1.
        self.Q = np.zeros((n_states, n_actions))
        self.Q[state1, goM] = 0.5
        self.T = np.zeros((n_states, n_actions, n_states))
        self.R = np.zeros((n_states, n_actions))
        

    
# %% functions

def SimulateTrial(Parameters, Estimates):
    state = state0
    while state != state5:
        if state == state1:
            IntegrateResults = ActionDistribution(state, Parameters, Estimates)
            state1_choice = IntegrateResults['Choice']
            action = IntegrateResults['Choice']
            
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
        
    return state1_choice
 
# %% load parameters

#Omegas = np.load('Omega only parameters.npy')

list_of_parameters = np.load('Best parameters.npy')

n_simulations = 100

for index, rat in enumerate(rats):
    #p = Parameters(omega = Omegas[index,0])
    p = Parameters(alpha = list_of_parameters[index, 0], beta = list_of_parameters[index, 1], u = list_of_parameters[index,2], omega = list_of_parameters[index,3])
    print(p)
    b = data[:,0] == rat
    rat_data = data[b,3]
    n_trials = len(rat_data)
    Simulation = np.zeros(( n_trials, n_simulations))
    
    for simulation in range(n_simulations):
        # initialise estimates
        
        estimates = Estimates()
        
        for t in range(n_trials):
            Simulation[t, simulation] = SimulateTrial(p, estimates)
            
    np.save('Rat' + str(int(rat)) + ' simulations', Simulation)