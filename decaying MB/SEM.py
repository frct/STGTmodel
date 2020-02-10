#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:29:02 2018

@author: francois
"""
import numpy as np

def sem(X):
    standard_deviation = np.std(X, axis=0)
    n_samples = X.shape[0]
    sem = standard_deviation / n_samples
    return sem