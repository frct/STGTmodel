#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:22:44 2018

histogram of probability difference scores for the mixed population

@author: francois
"""

from PCAIndex import IndexScore, NormalizedScore, SoftmaxScore
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

Distribution_name = 'Uniform distribution'

n_blocks = 10
size_blocks = 25
n_rats = 20

u_itis = np.array([0.01, 0.1])
histogram_colours = ['SkyBlue', 'IndianRed'] 
ITI_labels = ['Short', 'Long']

W = np.zeros((2,1))
p = np.zeros((2,1))
mu = np.zeros((2,1))
W_softmax = np.zeros((2,1))
p_softmax = np.zeros((2,1))
mu_softmax = np.zeros((2,1))

all_scores = np.zeros((n_rats * n_blocks, 2))
all_scores_softmax = np.zeros((n_rats * n_blocks, 2))

for iti, u_iti in enumerate(u_itis):

    score = np.zeros((n_rats, n_blocks))
    score_withoutexplore = np.zeros((n_rats, n_blocks))
    score_softmax1 = np.zeros((n_rats, n_blocks))
    score_softmax2 = np.zeros((n_rats, n_blocks))
    
    filename = 'Variable ITIs/' + Distribution_name + '/Mixed population simulations with intertrial ' + str(iti) + '.npz'
    data = np.load(filename)
    
    for rat in range(n_rats):
        for block in range(n_blocks):
            score[rat,block] = IndexScore(data['goL_counter'][rat, block], data['goM_counter'][rat, block], data['exp_counter'][rat, block])
            score_withoutexplore[rat,block] = NormalizedScore(data['goL_counter'][rat, block], data['goM_counter'][rat, block])
            
            block_distribution = data['Distribution'][block * size_blocks : (block + 1) * size_blocks, :, rat]
            average_exp = np.mean(block_distribution[:,0])
            average_goL = np.mean(block_distribution[:,1])
            average_goM = np.mean(block_distribution[:,2])
            
            score_softmax1[rat, block] = NormalizedScore(average_goL, average_goM)
            score_softmax2[rat, block] = SoftmaxScore(block_distribution[:,1], block_distribution[:,2])
            
    figure_filename = 'Variable ITIs/' + Distribution_name + '/Mixed population empirical probability difference histogram for ' + ITI_labels[iti] + '.eps'
    plt.figure(iti)
    plt.hist(score_withoutexplore.reshape(n_rats * n_blocks), edgecolor='black', color = histogram_colours[iti])
    plt.xlim(-1,1)
    plt.savefig(figure_filename)
    
    figure_filename = 'Variable ITIs/' + Distribution_name + '/Mixed population softmax probability difference histogram for ' + ITI_labels[iti] + '.eps'
    plt.figure(iti + 2)
    plt.hist(score_softmax2.reshape(n_rats * n_blocks), bins = np.linspace(-1, 1, num=13), edgecolor='black', color = histogram_colours[iti])
    plt.xlim(-1,1)
    plt.savefig(figure_filename)
    
    all_scores[:,iti] = score_withoutexplore.reshape(n_rats * n_blocks)
    W[iti], p[iti] = stats.wilcoxon(all_scores[:,iti])
    mu[iti] = np.mean(all_scores[:,iti])

    all_scores_softmax[:,iti] = score_softmax2.reshape(n_rats * n_blocks)
    W_softmax[iti], p_softmax[iti] = stats.wilcoxon(all_scores_softmax[:,iti])
    mu_softmax[iti] = np.mean(all_scores_softmax[:,iti])
    
W_directComparison, p_directComparison = stats.wilcoxon(all_scores[:,0],all_scores[:,1])
mu_directcomparison = np.mean(all_scores[:,1] - all_scores[:,0])

W_directComparison_softmax, p_directComparison_softmax = stats.wilcoxon(all_scores_softmax[:,0],all_scores[:,1])
mu_directcomparison_softmax = np.mean(all_scores_softmax[:,1] - all_scores_softmax[:,0])