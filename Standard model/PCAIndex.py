#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:22:02 2018

@author: francois
"""

import numpy as np

def IndexScore(goL, goM, exp):
    #print(ResponseBias)
    ProbabilityDifference = (goL - goM) / (goM + goL + exp)
    return ProbabilityDifference

def NormalizedScore(goL, goM):
    #print(ResponseBias)
    ProbabilityDifference = (goL - goM) / (goM + goL)
    return ProbabilityDifference

def SoftmaxScore(goL, goM):
    Score = np.mean((goL-goM)/(goM+goL))
    return Score