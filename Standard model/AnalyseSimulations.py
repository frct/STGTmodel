# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:25:20 2021

@author: franc
"""


import numpy as np
import matplotlib as plt
from scipy import stats

model = '4 parameter'

data = np.load('Binary classification of data.npy')

rats = np.unique(data[:,0])
#rats = np.array([5])
n_rats = len(rats)

n_sessions = 5
n_simulations = 100

st_trials = np.zeros((n_rats, n_sessions))
gt_trials = np.zeros((n_rats, n_sessions))

simulated_st_trials = np.zeros((n_rats, n_sessions, n_simulations))
simulated_gt_trials = np.zeros((n_rats, n_sessions, n_simulations))
y_simulated = np.zeros((n_rats, n_sessions))
y_data = np.zeros((n_rats, n_sessions))

for idx, rat in enumerate(rats):
    rat_data = data[data[:,0]==rat, :]
    Sessions = np.unique(rat_data[:, 1])
    #n_sessions =     len(Sessions)
    
    
    for idy, session in enumerate(Sessions):
        session_data = rat_data[rat_data[:,1] == session, :]
        st_trials[idx, idy] = sum(session_data[:,3] == 1)
        gt_trials[idx, idy] = sum(session_data[:,3] == 2)
    
    simulated_data = np.load(model + ' simulations/Rat' + str(int(rat)) + ' simulations.npy')
    
    n_simulations = len(simulated_data[0,:])
    
    for simulation in range(n_simulations):
        rat_data[:,3] = simulated_data[:, simulation]
        
        for idy, session in enumerate(Sessions):
            session_data = rat_data[rat_data[:,1] == session, :]
            simulated_st_trials[idx, idy, simulation] = sum(session_data[:,3] == 1)
            simulated_gt_trials[idx, idy, simulation] = sum(session_data[:,3] == 2)
            
    plt.pyplot.figure()
    y1 = np.min(simulated_st_trials[idx,:,:] / (simulated_st_trials[idx,:,:] + simulated_gt_trials[idx,:,:]), axis=1)
    y2 = np.max(simulated_st_trials[idx,:,:] / (simulated_st_trials[idx,:,:] + simulated_gt_trials[idx,:,:]), axis=1)
    y_simulated[idx,:] = np.transpose(np.mean(simulated_st_trials[idx,:,:] / (simulated_st_trials[idx,:,:] + simulated_gt_trials[idx,:,:]), axis=1))
    y_data[idx,:] = st_trials[idx,:] / (st_trials[idx,:]+gt_trials[idx,:])
    y = simulated_st_trials[idx,:,:] / (simulated_st_trials[idx,:,:] + simulated_gt_trials[idx,:,:])
    #y1 = np.quantile(simulated_st_trials[idx,:,:] / (simulated_st_trials[idx,:,:] + simulated_gt_trials[idx,:,:]), q = 0.25, axis=1)
    #y2 = np.quantile(simulated_st_trials[idx,:,:] / (simulated_st_trials[idx,:,:] + simulated_gt_trials[idx,:,:]), q = 0.75, axis=1)
    plt.pyplot.plot(range(n_sessions), y_data[idx,:])
    
    line, = plt.pyplot.plot(range(n_sessions), y_simulated[idx,:])
    plt.pyplot.fill_between(range(n_sessions), y1, y2, color=line.get_color(), alpha = 0.3)
    plt.pyplot.legend(('data', 'simulation'))
    plt.pyplot.plot(range(n_sessions), y, alpha = 0.1)
    plt.pyplot.savefig('rat ' + str(rat) + ' experimental data vs all simulations and shaded area.svg')
    
# plt.pyplot.figure()
# mean_simulation = np.mean(y_simulated, axis = 0)
# sem_simulation = stats.sem(y_simulated, axis = 0)
# mean_data = np.mean(y_data, axis = 0)
# sem_data = stats.sem(y_data, axis = 0)
# # plt.pyplot.plot(range(n_sessions), mean_simulation)
# # plt.pyplot.plot(range(n_sessions), mean_data)
# plt.pyplot.errorbar(range(n_sessions), mean_data, sem_data)
# plt.pyplot.errorbar(range(n_sessions), mean_simulation, sem_simulation)
# plt.pyplot.ylim((0, 0.8))
# plt.pyplot.legend(('data', 'simulation'))
# plt.pyplot.savefig(model + ' average simulation.svg')