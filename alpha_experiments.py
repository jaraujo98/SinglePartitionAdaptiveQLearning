#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:04:27 2020

This file implements linear, exponential, and periodical (sinusoidal) schedules
for the temperature in Boltzmann exploration. It was used in an ealier version
of SPAQL.

Run it to see an example of how the probabilities of picking an action evolve
for each schedule.
"""
import numpy as np

eps = 1e-2

def get_linear_alpha(c, I, M):
    """
    c: current iteration
    I: maximum number of iterations
    M: maximum alpha
    """
    alpha = 1/I
    ret = M * (1 - alpha*c)
    if ret < eps:
        ret = eps
    return ret

def get_exp_alpha(c, I, M):
    """
    c: current iteration
    I: maximum number of iterations
    M: maximum alpha
    """
    alpha = np.log(M + 1)/I
    ret = np.exp(alpha*(I-c)) - 1
    if ret < eps:
        ret = eps
    return ret

def get_sin_alpha(c, I, M, periods=2):
    """
    c: current iteration
    I: maximum number of iterations
    M: maximum alpha
    periods: number of times actions are uniformly sampled
    """
    alpha = np.pi/I * (2*periods - 1)
    ret = M * (1 - np.cos(alpha*(I-c)))/2
    if ret < eps:
        ret = eps
    return ret

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    nIters = 500
    max_alpha = 5
    
    labels = [f"Action {i}" for i in range(1, 6)]
    
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    ax1 = ax[0]
    
    iteration = np.arange(0, nIters)
    q = 5 * np.random.random((1,5))
    y = np.zeros((iteration.size, q.size))
    for a in iteration:
        alpha = get_linear_alpha(a, nIters/2, max_alpha)
        q1 = np.exp(q/alpha)
        y[a,:] = q1/np.sum(q1)
        
    ax1.stackplot(iteration, np.transpose(y), labels=labels)
    ax1.set_xlabel(r"Training iteration")
    ax1.set_ylabel("P(action)")
    ax1.legend(loc='upper right')
    ax1.set_title("Linear schedule")
    
    ax2 = ax[1]
    
    y = np.zeros((iteration.size, q.size))
    for a in iteration:
        alpha = get_exp_alpha(a, nIters/2, max_alpha)
        q1 = np.exp(q/alpha)
        y[a,:] = q1/np.sum(q1)
        
    ax2.stackplot(iteration, np.transpose(y), labels=labels)
    ax2.set_xlabel("Training iteration")
    ax2.set_ylabel("P(action)")
    ax2.legend(loc='upper right')
    ax2.set_title("Exponential schedule")
    
    ax3 = ax[2]
    
    y = np.zeros((iteration.size, q.size))
    for a in iteration:
        alpha = get_sin_alpha(a, nIters/2, max_alpha, periods=1)
        q1 = np.exp(q/alpha)
        y[a,:] = q1/np.sum(q1)
        
    ax3.stackplot(iteration, np.transpose(y), labels=labels)
    ax3.set_xlabel("Training iteration")
    ax3.set_ylabel("P(action)")
    ax3.legend(loc='upper right')
    ax3.set_title("Sinusoidal schedule, 1 period")
    
    ax3 = ax[3]
    
    y = np.zeros((iteration.size, q.size))
    for a in iteration:
        alpha = get_sin_alpha(a, nIters/2, max_alpha, periods=3)
        q1 = np.exp(q/alpha)
        y[a,:] = q1/np.sum(q1)
        
    ax3.stackplot(iteration, np.transpose(y), labels=labels)
    ax3.set_xlabel("Training iteration")
    ax3.set_ylabel("P(action)")
    ax3.legend(loc='upper right')
    ax3.set_title("Sinusoidal schedule, 3 periods")
    
    plt.tight_layout()
    plt.show()