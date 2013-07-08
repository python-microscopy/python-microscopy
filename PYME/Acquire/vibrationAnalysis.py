#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from PYME.Analysis import MetaData
from PYME.Analysis.FitFactories import ConfocCOIR
import numpy as np
from PYME.DSView import View3D

class VibrAnal:
    def __init__(self, scope, threshold=100):
        self.scope = scope
        self.mdh = MetaData.TIRFDefault
        self.threshold = threshold
        
        self.mdh['tIndex'] = 0
        self.mdh['Camera.ADOffset'] = scope.pa.dsa.min()
        self.i = 0
        self.dt = np.zeros(512, ConfocCOIR.FitResultsDType)
        
        self.fx = np.zeros(256)
        self.x = self.dt['fitResults']['x0']
        
        self.vx = View3D(self.x, mode='fgraph')
        self.vfx = View3D(self.fx, mode='fgraph')
        
        scope.pa.WantFrameNotification.append(self.frameCOI)
        scope.pa.WantFrameGroupNotification.append(self.OnFrameGroup)
        
    def frameCOI(self, caller):  
        self.dt[self.i] = ConfocCOIR.FitFactory(self.scope.pa.dsa, self.mdh, self.threshold)
        self.i +=1
        self.i %= 512
        
    def OnFrameGroup(self, caller):
        self.fx[:] = np.log10(np.abs(np.fft.fft(self.x - self.x.mean())[:256]))
        self.vx.do.OnChange()
        self.vfx.do.OnChange()
        
    def Detach(self):
        self.scope.pa.WantFrameGroupNotification.remove(self.OnFrameGroup)
        self.scope.pa.WantFrameNotification.remove(self.frameCOI)
        
    