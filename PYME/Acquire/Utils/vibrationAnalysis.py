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
from PYME.localization.FitFactories import ConfocCOIR
import numpy as np
from PYME.DSView import View3D

class VibrAnal:
    def __init__(self, scope, threshold=100):
        self.scope = scope
        self.mdh = MetaData.TIRFDefault
        self.threshold = threshold
        
        self.mdh['tIndex'] = 0
        self.mdh['Camera.ADOffset'] = scope.frameWrangler.currentFrame.min()
        self.i = 0
        self.dt = np.zeros(512, ConfocCOIR.FitResultsDType)
        
        self.fx = np.zeros(256)
        self.x = self.dt['fitResults']['x0']
        
        self.fy = np.zeros(256)
        self.y = self.dt['fitResults']['y0']
        
        self.vx = View3D(self.x, mode='fgraph')
        self.vfx = View3D(self.fx, mode='fgraph')
        self.vy = View3D(self.y, mode='fgraph')
        self.vfy = View3D(self.fy, mode='fgraph')
        
        #scope.frameWrangler.WantFrameNotification.append(self.frameCOI)
        #scope.frameWrangler.WantFrameGroupNotification.append(self.OnFrameGroup)
        self.scope.frameWrangler.onFrame.connect(self.frameCOI)
        self.scope.frameWrangler.onFrameGroup.connect(self.OnFrameGroup)
        
    def frameCOI(self, **kwargs):  
        self.dt[self.i] = ConfocCOIR.FitFactory(self.scope.frameWrangler.currentFrame, self.mdh, self.threshold)
        self.i +=1
        self.i %= 512
        if self.i == 0:
            self.fx[:] = np.maximum(np.log10(np.abs(np.fft.fft(self.x - self.x.mean())[:256])), 0)
            self.fy[:] = np.maximum(np.log10(np.abs(np.fft.fft(self.y - self.y.mean())[:256])), 0)
        
    def OnFrameGroup(self, **kwargs):
        
        self.vx.do.OnChange()
        self.vfx.do.OnChange()
        self.vy.do.OnChange()
        self.vfy.do.OnChange()
        
    def Detach(self):
        #self.scope.frameWrangler.WantFrameGroupNotification.remove(self.OnFrameGroup)
        #self.scope.frameWrangler.WantFrameNotification.remove(self.frameCOI)
        self.scope.frameWrangler.onFrame.disconnect(self.frameCOI)
        self.scope.frameWrangler.onFrameGroup.disconnect(self.OnFrameGroup)
        
    
