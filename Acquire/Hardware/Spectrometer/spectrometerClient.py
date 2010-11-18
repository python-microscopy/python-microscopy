#!/usr/bin/python

##################
# SpectrometerClient.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Pyro.core
import numpy as np

class SpecClient: #for client machine
    def __init__(self, chan=0):
        self.chan = chan
        self.wrapper = Pyro.core.getProxyForURI('PYRONAME://USB2000p')
        self.wrapper.setCorrectForDetectorNonlinearity(chan, 1)
        self.wrapper.setCorrectForElectricalDark(chan, 1)

    def getWavelengths(self):
        return np.array(self.wrapper.getWavelengths(self.chan)).view('>f8')

    def getSpectrum(self):
        return np.array(self.wrapper.getSpectrum(self.chan)).view('>f8')

    def setIntegrationTime(self, itime):
        self.wrapper.setIntegrationTime(self.chan,int(itime*1e3))
