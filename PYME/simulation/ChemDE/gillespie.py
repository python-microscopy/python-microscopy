# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:23:01 2014

@author: david
"""

import numpy as np

N_AVAGADRO = 6.023e23

class gillespie(object):
    def __init__(self, equationSystem, volume=1.0):
        self.S = equationSystem
        #self.state = 1.0*initParams
        self.volume = float(volume)
        #self.t = 0
        
        self.n_to_conc = 1.0/(N_AVAGADRO*self.volume)
        
        self.ns = len(self.S.species)
        self.nr = len(self.S.reactions)
        
        self.ratePow = np.zeros([self.nr, self.ns])
        self.molDiff = np.zeros_like(self.ratePow)
        self.rates = np.zeros(self.nr)
        
        for i, reaction in enumerate(self.S.reactions):
            n_f, n_b, m_f, m_b, k_f, k_b = reaction.getRateInfo(self.S.species, self.S.ties, self.S.constants)
            
            self.ratePow[i, :] = m_f 
            #self.ratePow[i+self.nr, :] = m_b
            
            self.molDiff[i, :] = n_f
            #self.molDiff[i+self.nr, :] = n_b
            
            self.rates[i] = 1e6*k_f
            #self.rates[i+self.nr] = 1e6*k_b
            
        #self.ts = [0]
        #self.stateHistory = [1.0*self.state]

        
    def _draw(self):
        #print np.exp((np.log((self.state + .001)*self.n_to_conc)[None,:]*self.ratePow).sum(1)).shape
        kl = np.exp((np.log((self.state + .001)*self.n_to_conc)[None,:]*self.ratePow).sum(1))*self.rates*((self.state[None,:] -self.ratePow)>=0).prod(1)
        
        k_s = kl.sum()
        #print k_s, kl.shape
        
        #choose a time point based on total rate
        dt = np.random.exponential(1./k_s)
        
        #decide which reaction to take
        ri = np.cumsum(kl/k_s).searchsorted(np.random.rand())
        
        #print ri, self.molDiff[ri, :]
        
        self.state = self.state + self.molDiff[ri, :]
        self.t = self.t + dt
        
        self.stateHistory.append(self.state)
        self.ts.append(self.t)
        
    def integrate(self, initParams, maxTime=3600, maxIters = 50e3):
        self.state = 1.0*initParams
        self.t = 0
        
        self.ts = [0]
        self.stateHistory = [1.0*self.state]

        nIters = 0        
        
        while (self.t <  maxTime) and (nIters < maxIters):
            self._draw()
            nIters += 1
            
        return self.ts, np.array(self.stateHistory)
            
        