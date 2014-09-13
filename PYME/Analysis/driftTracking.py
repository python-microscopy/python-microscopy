# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:02:50 2014

@author: David Baddeley
"""

import numpy as np
from pylab import fftn, ifftn, fftshift, ifftshift
import time
from scipy import ndimage

def correlateFrames(A, B):
    A = A.squeeze()/A.mean() - 1
    B = B.squeeze()/B.mean() - 1
    
    X, Y = np.mgrid[0.0:A.shape[0], 0.0:A.shape[1]]
    
    C = ifftshift(np.abs(ifftn(fftn(A)*ifftn(B))))
    
    Cm = C.max()    
    
    Cp = np.maximum(C - 0.5*Cm, 0)
    Cpsum = Cp.sum()
    
    x0 = (X*Cp).sum()/Cpsum
    y0 = (Y*Cp).sum()/Cpsum
    
    return x0 - A.shape[0]/2, y0 - A.shape[1]/2, Cm, Cpsum
    
    
def correlateAndCompareFrames(A, B):
    A = A.squeeze()/A.mean() - 1
    B = B.squeeze()/B.mean() - 1
    
    X, Y = np.mgrid[0.0:A.shape[0], 0.0:A.shape[1]]
    
    C = ifftshift(np.abs(ifftn(fftn(A)*ifftn(B))))
    
    Cm = C.max()    
    
    Cp = np.maximum(C - 0.5*Cm, 0)
    Cpsum = Cp.sum()
    
    x0 = (X*Cp).sum()/Cpsum
    y0 = (Y*Cp).sum()/Cpsum
    
    dx, dy = x0 - A.shape[0]/2, y0 - A.shape[1]/2
    
    As = ndimage.shift(A, [-dx, -dy])
    
    #print A.shape, As.shape
    
    return (As -B).mean(), dx, dy
    
    
class correlator(object):
    def __init__(self, imageSource):
        self.d = imageSource
        
        self.X, self.Y = np.mgrid[0.0:self.d.shape[0], 0.0:self.d.shape[1]]
        self.X -= self.d.shape[0]/2
        self.Y -= self.d.shape[1]/2

        
    def setRefA(self):
        self.refA = (1.0*self.d).squeeze()/self.d.mean() - 1        
        self.FA = ifftn(self.refA)
        
    def setRefB(self):
        self.refB = (1.0*self.d).squeeze()/self.d.mean() - 1        
        
    def setRefC(self):
        self.refC = (1.0*self.d).squeeze()/self.d.mean() - 1        
        
        
    def compare(self):
        dm = self.d.squeeze()/self.d.mean() - 1
        
        C = ifftshift(np.abs(ifftn(fftn(dm)*self.FA)))
        
        Cm = C.max()    
        
        Cp = np.maximum(C - 0.5*Cm, 0)
        Cpsum = Cp.sum()
        
        dx = (self.X*Cp).sum()/Cpsum
        dy = (self.Y*Cp).sum()/Cpsum
        
        ds = ndimage.shift(dm, [-dx, -dy])
        
        #print A.shape, As.shape
        
        return abs(ds -self.refA).mean(), abs(ds -self.refB).mean(), abs(ds -self.refC).mean(), dx, dy
        
    def setRefs(self, piezo):
        time.sleep(0.5)
        p = piezo.GetPos()
        self.setRefA()
        piezo.MoveTo(0, p -.2)
        time.sleep(0.5)
        self.setRefB()
        piezo.MoveTo(0,p +.2)
        time.sleep(0.5)
        self.setRefC()
        piezo.MoveTo(0, p)
    
    
    