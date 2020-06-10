#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:25:32 2018

@author: francois
"""


import numpy as np

#defining the different indices of states, actions and features

a= float('nan') # for any undefined variable

#states

state0 = 0
state1 = 1
state2 = 2
state3 = 3
state4 = 4
state5 = 5
state6 = 6
state7 = 7
n_states = 8

#actions

goE = 0
goL = 1
goM = 2
eng = 3
wait = 4
eat = 5
n_actions = 6

# features: Nothing, Lever, Magazine and Food

N = 0
L = 1
M = 2
F = 3
n_features = 4

# mapping of state x action to features

StatexAction_to_Feature = np.array([[N, a, a, a, a, a],
                  [N, L, M, a, a, a],
                  [a, a, a, L, a, a],
                  [a, a, a, a, N, a],
                  [a, a, a, M, a, a],
                  [a, a, F, a, a, a],
                  [a, a, F, a, a, a],
                  [a, a, a, a, a, M]])

# mapping of state x action to next state
StatexAction_to_State = np.array([[state1,  a,  a,  a,  a,  a], 
                  [state3, state2, state4,  a,  a,  a], 
                  [ a,  a,  a, state5,  a,  a],
                  [ a,  a,  a,  a, state6,  a],
                  [ a,  a,  a, state7,  a,  a],
                  [ a,  a, state7,  a,  a,  a],
                  [ a,  a, state7,  a,  a,  a], 
                  [ a,  a,  a,  a,  a, state0]])

# mapping of state x action to reward: only eating in state 7 gives a reward
StatexAction_to_Reward = np.zeros((n_states,n_actions))
StatexAction_to_Reward[state7,eat] = 1.;