# -*- coding: utf-8 -*-

"""
Created on Mon August 12 2020

@author: zacsimile
"""

from PYME.Acquire.Hardware.pco import pco_cam

noiseProperties = {
'61003940' : {
        'ReadNoise': 1.3, 
        'ElectronsPerCount': 0.46,
        'NGainStages': 0,
        'ADOffset': 100,
        'DefaultEMGain': 1,
        'SaturationThreshold': (2**16 - 1)
        },
}

class PcoEdge42LT(pco_cam.PcoCam):
    def __init__(self, camNum, debuglevel='off'):
        pco_cam.PcoCam.__init__(self, camNum, debuglevel)

    def Init(self):
        pco_cam.PcoCam.Init(self)
        self.noiseProps = noiseProperties[self.GetSerialNumber()]