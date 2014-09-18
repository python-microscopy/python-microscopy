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
    def __init__(self, scope, piezo=None):
        self.scope = scope
        self.piezo = piezo
        
        self.focusTolerance = .05 #how far focus can drift before we correct
        self.deltaZ = 0.2 #z increment used for calibration
        #self.initialise()
        
    def initialise(self):
        d = 1.0*self.scope.pa.dsa.squeeze()        
        
        self.X, self.Y = np.mgrid[0.0:d.shape[0], 0.0:d.shape[1]]
        self.X -= d.shape[0]/2
        self.Y -= d.shape[1]/2
        
        #we want to discard edges after accounting for x-y drift
        self.mask = np.ones_like(d)
        self.mask[:10, :] = 0
        self.mask[-10:, :] = 0
        self.mask[:, :10] = 0
        self.mask[:,-10:] = 0
        
        self.calibState = 0 #completely uncalibrated

        
        self.lockFocus = False
        self.lastAdjustment = 5 
        
        
        self.history = []
        
        self.historyCorrections = []

        
    def setRefA(self):
        d = 1.0*self.scope.pa.dsa.squeeze()
        self.refA = d/d.mean() - 1        
        self.FA = ifftn(self.refA)
        self.refA *= self.mask
        
    def setRefB(self):
        d = 1.0*self.scope.pa.dsa.squeeze()
        self.refB = d/d.mean() - 1
        self.refB *= self.mask        
        
    def setRefC(self):
        d = 1.0*self.scope.pa.dsa.squeeze()
        self.refC = d/d.mean() - 1
        self.refC *= self.mask
        
        self.dz = (self.refC - self.refB).ravel()
        self.dzn = 2./np.dot(self.dz, self.dz)
        
    #def setRefD(self):
    #    self.refD = (1.0*self.d).squeeze()/self.d.mean() - 1 
    #    self.refD *= self.mask
        
        #self.dz = (self.refC - self.refA).ravel()
        
        
    def compare(self):
        d = 1.0*self.scope.pa.dsa.squeeze()
        dm = d/d.mean() - 1
        
        #find x-y drift
        C = ifftshift(np.abs(ifftn(fftn(dm)*self.FA)))
        
        Cm = C.max()    
        
        Cp = np.maximum(C - 0.5*Cm, 0)
        Cpsum = Cp.sum()
        
        dx = (self.X*Cp).sum()/Cpsum
        dy = (self.Y*Cp).sum()/Cpsum
        
        ds = ndimage.shift(dm, [-dx, -dy])*self.mask
        
        #print A.shape, As.shape
        
        ds_A = (ds - self.refA)
        
        return dx, dy, self.deltaZ*np.dot(ds_A.ravel(), self.dz)*self.dzn
        
    
    def tick(self, caller=None):
        if not 'mask' in dir(self) or not self.scope.pa.dsa.shape[:2] == self.mask.shape[:2]:
            self.initialise()
            
        #called on a new frame becoming available
        if self.calibState == 0:
            self.piezo.SetOffset(0)
            self.calibState = 1
        elif self.calibState == 1:
            self.calibState = 2
        elif self.calibState == 2:
            self.setRefA()
            self.piezo.SetOffset(-self.deltaZ)
            self.calibState = 3
        elif self.calibState == 3:
            self.calibState = 4
        elif self.calibState == 4:
            self.setRefB()
            self.piezo.SetOffset(self.deltaZ)
            self.calibState = 5
        elif self.calibState == 5:
            self.calibState = 6
        elif self.calibState == 6:
            self.setRefC()
            self.piezo.SetOffset(0)
            self.calibState = 7
        elif self.calibState == 7:
            #fully calibrated
            dx, dy, dz = self.compare()
            
            #print dx, dy, dz
            
            self.history.append((time.time(), dx, dy, dz))
            
            if self.lockFocus:
                if abs(dz) > self.focusTolerance and self.lastAdjustment >= 2:
                    self.piezo.SetOffset(self.piezo.GetOffset() - dz)
                    self.historyCorrections.append((time.time(), dz))
                    self.lastAdjustment = 0
                else:
                    self.lastAdjustment += 1
                    
            
    def reCalibrate(self):
        self.calibState = 0
        
    def register(self):
        self.scope.pa.WantFrameGroupNotification.append(self.tick)
        
    def deregister(self):
        self.scope.pa.WantFrameGroupNotification.remove(self.tick)
    
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
    
    
    