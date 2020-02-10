#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:22:13 2018

histogram of probability difference scores for either FMF or MB only model

@author: francois
"""

from PCAIndex import IndexScore
import numpy as np
import matplotlib.pyplot as plt

phenotype = 'ST'

n_blocks = 10
n_rats = 20
PCA_blocks = 2 #number of blocks starting from the last one involved in the calculation of PCA index

u_itis = np.array([0, 0.01, 0.02])
histogram_colours = ['SkyBlue', 'Plum', 'IndianRed'] 
ITI_labels = ['Short', 'Intermediate', 'Long']

for iti, u_iti in enumerate(u_itis):

    score = np.zeros((n_rats, n_blocks))
    
    filename = 'Variable ITIs/' + phenotype + 'Simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    
    for rat in range(n_rats):
        for block in range(n_blocks):
            score[rat,block] = IndexScore(data['goL_counter'][rat, block], data['goM_counter'][rat, block], data['exp_counter'][rat, block])
    
    figure_filename = 'Variable ITIs/' + phenotype + ' probability difference histogram for ' + ITI_labels[iti] + ' ITI.eps'
    plt.figure(iti)
    plt.hist(score.reshape(n_rats * n_blocks), edgecolor='black', color = histogram_colours[iti])
    plt.xlim(-1,1)
    plt.savefig(figure_filename)