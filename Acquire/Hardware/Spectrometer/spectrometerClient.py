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
    def __init__(self):
        self.wrapper = Pyro.core.getProxyForURI('PYRONAME://USB2000p')

    def getWavelengths(self, chan=0):
        return np.array(self.wrapper.getWavelengths(chan)).view('>f8')

    def getSpectrum(self, chan=0):
        return np.array(self.wrapper.getSpectrum(chan)).view('>f8')

    def setIntegrationTime(self, itime, chan=0):
        self.wrapper.setIntegrationTime(chan,int(itime*1e3))
