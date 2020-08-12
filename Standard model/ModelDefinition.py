# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:23:20 2020

@author: franc
"""


import numpy as np
from TaskVariables import StatexAction_to_Feature, StatexAction_to_State, StatexAction_to_Reward, state7, state1, n_actions, F

def Feature(State, Action):
    """ returns feature for a given state and action """
    return int(StatexAction_to_Feature[State, Action])

def PossibleFeatures(State):
    """ returns the possible features corresponding to a given state """
    f = StatexAction_to_Feature[State, :]
    f = f[~np.isnan(f)]
    return f.astype(int)

def NextState(State, Action):
    """ returns next state for a given action * state """
    return int(StatexAction_to_State[State, Action])

def PossibleActions(State):
    """ returns the possible actions in a given state """
    f = StatexAction_to_State[State, :]
    actions = np.argwhere(~np.isnan(f))
    return actions

def FMF_component(State, Action, Value, Parameters):
    """ given a (state, action) pair, computes Feature-Model_Free update of 
    the value function V, and returns the updated values and the RPE, or 
    dopaminergic signal """
    c = Feature(State, Action)
    r = StatexAction_to_Reward[State, Action]
    if State != state7 and State != state1 :
        next_features = PossibleFeatures(NextState(State, Action))
        delta = r + Parameters.gamma * max(Value[next_features]) - Value[c]
        Value[c] = Value[c] + Parameters.alpha * delta
    else :
        # we compute an RPE for comparison with dopamine signal, but do not update the value of food which is supposed constant
        delta = r - Value[c]
    Value[F] = 1
    return {'Value' : Value, 'DA' : delta}


def MB_component(State, Action, Estimates, Parameters):
    """ given a (state, action) pair, computes the model-based updates of the 
    Q-function, transition function and reward function which it returns """
    r = StatexAction_to_Reward[State, Action]
    Estimates.R[State, Action] = Estimates.R[State, Action] + Parameters.alpha * (r - Estimates.R[State, Action])
    if State != state7: # task is episodic, transition from last state to state 0 must be treated differently
        next_state = NextState(State, Action)
        Estimates.T[State, Action, :] = (1 - Parameters.alpha) * Estimates.T[State, Action, :]
        Estimates.T[State, Action, next_state] += Parameters.alpha
        maxQ = np.amax(Estimates.Q, axis = 1)
        Estimates.Q[State, Action] = Estimates.R[State, Action] + Parameters.gamma * np.dot(Estimates.T[State, Action, :], maxQ)
    else :
        Estimates.Q[State, Action] = Estimates.R[State, Action]
    #print('Estimates' +str(Estimates.Q))
    return Estimates


def ActionDistribution(State, Parameters, Estimates):
    """ combines the outcomes of the FMF and MB modules to determine the 
    combined P-value of each possible action, as well as its probability of 
    selection and randomly selects an action based on these probabilities """
    actions = PossibleActions(State)
    n_possibleactions = len(actions)
    P = np.zeros(n_possibleactions)
    for index, action in enumerate(actions):
        c = Feature(State, action)
        Advantage = Estimates.Q[State, action] - max(Estimates.Q[State, :])
        P[index] = (1 - Parameters.omega) * Advantage + Parameters.omega * Estimates.V[c]
        
    distribution = np.zeros(n_possibleactions)
    #print('P' + str(P))
    for i in range(n_possibleactions):
        distribution[i] = np.exp(P[i] * Parameters.beta) / sum(np.exp(P[:] * Parameters.beta))
        #print('numerator' + str(np.exp(P[i] * Parameters.beta)))
        #print('denominator' + str(sum(np.exp(P[:] * Parameters.beta))))
        
    #print('Distrubution:' +str(distribution))
    j = np.random.choice(n_possibleactions, 1, p=distribution)
    choice = actions[j]
    return {'Action probabilities': distribution, 'Choice': choice, 'Pvalues': P}