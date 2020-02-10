#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:37:30 2018

@author: francois
"""

import numpy as np


a= float('nan') # for any undefined variable

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

wait = 0
goL = 1
goM = 2
eat = 3
n_actions = 4

## features: nothing, Lever, Magazine and Food

E = 0
L = 1
M = 2
F = 3
n_features = 4

# mapping of state x action to features

SxA_F = np.array([[E, a, a, a],
                  [a, L, M, a],
                  [L, a, M, a],
                  [M, L, a, a],
                  [L, a, a, a],
                  [M, a, a, a],
                  [a, a, F, a],
                  [a, a, a, F]])

# mapping of state x action to next state
SxA_S = np.array([[s1,  a,  a,  a], 
                  [ a, s2, s3,  a],
                  [s4,  a, s5,  a],
                  [s5, s4,  a,  a],
                  [s6,  a,  a,  a],
                  [s7,  a,  a,  a],
                  [ a,  a, s7,  a],
                  [ a,  a,  a, s0]])

#mapping of state x action to reward
SxA_R = np.zeros((n_states,n_actions))
SxA_R[s7,eat] = 1.;