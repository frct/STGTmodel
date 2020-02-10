#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:10:55 2018

@author: francois

new model with no explore environment option, and two features for the food cup: empty or full
"""

import numpy as np

#defining the different indices of states, actions and features

a= float('nan') # for any undefined variable

#states

s0 = 0
s1 = 1
s2 = 2
s3 = 3
s4 = 4
s5 = 5
n_states = 6

#actions

wait = 0
goL = 1
goM = 2
eng = 3
eat = 4
n_actions = 5

# features: environment, Lever, Magazine full or empty and Reward

O = 0
L = 1
M = 2
R = 3
n_features = 4

# mapping of state x action to features

SxA_F = np.array([[O, a, a, a, a],
                  [a, L, M, a, a],
                  [a, a, a, L, a],
                  [a, L, a, M, a],
                  [a, a, R, a, a],
                  [a, a, a, a, R]])

# mapping of state x action to next state
SxA_S = np.array([[s1,  a,  a,  a,  a], 
                  [ a, s2, s3,  a,  a], 
                  [ a,  a,  a, s4,  a],
                  [ a, s4,  a, s5,  a],
                  [ a,  a, s5,  a,  a],
                  [ a,  a,  a,  a, s0]])

#mapping of state x action to reward
SxA_R = np.zeros((n_states,n_actions))
SxA_R[s5,eat] = 1.;