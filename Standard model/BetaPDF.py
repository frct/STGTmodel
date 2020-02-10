#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:35:04 2018

plot pdf of beta distribution

@author: francois
"""

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)

a, b = 8, 4

x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 100)

ax.plot(x, beta.pdf(x, a, b), 'r-', lw=2, alpha=0.6, label='beta pdf')

plt.savefig('Variable ITIs/Beta distribution/probability density function.eps')