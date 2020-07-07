#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:22:47 2020

This file implements wrapper classes for the oil and ambulance problems.
It was a simple way to get joblib running without throwing errors...
"""

import numpy as np
from src import environment

class OilLaplace(environment.OilEnvironment):
    def __init__(self, epLen, starting_state, lam):
        self.epLen = epLen
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        self.lam = lam
        
    def oil_prob(self, x):
        return np.exp(-1*self.lam*np.abs(x - 0.7-np.pi/60))
    
class OilQuadratic(environment.OilEnvironment):
    def __init__(self, epLen, starting_state, lam):
        self.epLen = epLen
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        self.lam = lam
        
    def oil_prob(self, x):
        return 1 - self.lam*(x-0.7-np.pi/60)**2
    
class Ambulance(environment.AmbulanceEnvironment):
    def __init__(self, epLen, arrivals, alpha, starting_state):
        self.epLen = epLen
        self.arrivals = beta
        self.alpha = alpha
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        
def beta(a=5, b=2):
    return np.random.beta(a, b)

def uniform(a=0, b=1):
    return np.random.uniform(a, b)