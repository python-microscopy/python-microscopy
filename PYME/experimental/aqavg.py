# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:00:16 2014

@author: David Baddeley
"""

import numpy as np

class FrameAverager(object):
    def __init__(self, pa, nAvg=10):
        self.nAvg = nAvg
        self.pa = pa
        
        self.nCurrent = 0        
        self.frameBuff = []
        self.frameSum = np.zeros(pa.dsa.shape, 'float64')
        self.frameAvg = np.zeros(pa.dsa.shape, 'float64')
        
        self.pa.WantFrameNotification.append(self.tick)
        
    def tick(self, caller=None):
        if not self.frameSum.shape == self.pa.dsa.shape:
            self.frameSum = np.zeros(self.pa.dsa.shape, 'float64')
            self.frameAvg = np.zeros(self.pa.dsa.shape, 'float64')
            self.nCurrent = 0
            self.frameBuff.clear()
        
        if self.nCurrent >= self.nAvg:
            ff = self.frameBuff.pop(0)
            self.frameSum -= ff
            self.nCurrent -= 1
            
        self.frameBuff.append(1.0*self.pa.dsa)
        self.frameSum += self.frameBuff[-1]
        self.nCurrent += 1
        
        self.frameAvg[:] = (self.frameSum/self.nCurrent)[:]
        
    