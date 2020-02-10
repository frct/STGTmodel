#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:05:07 2018

@author: francois
"""

from PCAIndex import IndexScore
import numpy as np
import matplotlib.pyplot as plt

phenotypes = ['GT', 'IG', 'ST']
n_phenotypes = len(phenotypes)
pheno_colours = ['r', 'b', 'k']
n_blocks = 8
n_rats = 14
magazine_present = True
lever_present = True
PCA_blocks = 2 #number of blocks starting from the last one involved in the calculation of PCA index

if magazine_present == False:
    ITIcondition = 'magazine absent'
elif lever_present:
    ITIcondition = 'lever present'
else:
    ITIcondition = 'lever absent'
    
iti_scale = 0.5

score = np.zeros((n_rats, n_blocks, n_phenotypes))
PCA = np.zeros((n_rats, n_phenotypes))

for counter, phenotype in enumerate(phenotypes):
    filename = ITIcondition + '/' + phenotype + 'Simulations with intertrial ' + str(iti_scale) + '.npz'
    data = np.load(filename)
    for rat in range(n_rats):
        for b in range(n_blocks):
            score[rat,b,counter] = IndexScore(data['goL_counter'][rat, b], data['goM_counter'][rat, b], data['exp_counter'][rat, b])
        PCA[rat, counter] = np.mean(score[rat, n_blocks-PCA_blocks : n_blocks, counter])

    plt.figure(1)
    plt.scatter(range(n_rats * counter, n_rats * (counter + 1)), PCA[:,counter], c=pheno_colours[counter])

