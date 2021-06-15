# -*- coding: utf-8 -*-

"""
Created on Mon August 12 2020

@author: zacsimile
"""

# from PYME.Acquire.Hardware.pco import pco_cam
from PYME.Acquire.Hardware.pco import pco_sdk_cam

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

class PcoEdge42LT(pco_sdk_cam.PcoSdkCam):
    def __init__(self, camNum, debuglevel='off'):
        pco_sdk_cam.PcoSdkCam.__init__(self, camNum, debuglevel)

    def Init(self):
        pco_sdk_cam.PcoSdkCam.Init(self)
        self.noiseProps = noiseProperties[self.GetSerialNumber()]

    def GenStartMetadata(self, mdh):
        pco_sdk_cam.PcoSdkCam.GenStartMetadata(self, mdh)
        if self.active:
            mdh.setEntry('Camera.ADOffset', self.noiseProps['ADOffset'])
