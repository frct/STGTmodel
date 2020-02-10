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

s0 = 0
s1 = 1
s2 = 2
s3 = 3
s4 = 4
s5 = 5
s6 = 6
s7 = 7
n_states = 8

#actions

goE = 0
goL = 1
goM = 2
eng = 3
wait = 4
eat = 5
n_actions = 6

# features: nothing, Lever, Magazine and Food

E = 0
L = 1
M = 2
F = 3
n_features = 4

# mapping of state x action to features

SxA_F = np.array([[E, a, a, a, a, a],
                  [E, L, M, a, a, a],
                  [a, a, a, L, a, a],
                  [a, a, a, a, E, a],
                  [a, a, a, M, a, a],
                  [a, a, F, a, a, a],
                  [a, a, F, a, a, a],
                  [a, a, a, a, a, M]])

# mapping of state x action to next state
SxA_S = np.array([[s1,  a,  a,  a,  a,  a], 
                  [s3, s2, s4,  a,  a,  a], 
                  [ a,  a,  a, s5,  a,  a],
                  [ a,  a,  a,  a, s6,  a],
                  [ a,  a,  a, s7,  a,  a],
                  [ a,  a, s7,  a,  a,  a],
                  [ a,  a, s7,  a,  a,  a], 
                  [ a,  a,  a,  a,  a, s0]])

#mapping of state x action to reward
SxA_R = np.zeros((n_states,n_actions))
SxA_R[s7,eat] = 1.;