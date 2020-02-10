#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:22:02 2018

@author: francois
"""


def IndexScore(goL, goM, exp):
    if goL + goM != 0:
        ResponseBias = (goL - goM) / (goL + goM)
    else:
        ResponseBias = 0
        print('only exploration!')
    #print(ResponseBias)
    ProbabilityDifference = (goL - goM) / (goM + goL + exp)
    Score = (ResponseBias + ProbabilityDifference) / 2
    return Score