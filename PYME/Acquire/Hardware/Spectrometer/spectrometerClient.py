#!/usr/bin/python

##################
# SpectrometerClient.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import Pyro.core
import numpy as np

class SpecClient: #for client machine
    def __init__(self, chan=0):
        self.chan = chan
        self.wrapper = Pyro.core.getProxyForURI('PYRONAME://USB2000p')
        self.setCorrectForDetectorNonlinearity(1)
        self.setCorrectForElectricalDark(1)

    def getWavelengths(self):
        return np.array(self.wrapper.getWavelengths(self.chan)).view('>f8')

    def getSpectrum(self):
        return np.array(self.wrapper.getSpectrum(self.chan)).view('>f8')

    def setIntegrationTime(self, itime):
        self.wrapper.setIntegrationTime(self.chan,int(itime*1e3))

    def getIntegrationTime(self):
        return self.wrapper.getIntegrationTime(self.chan)

    def setScansToAverage(self, nAvg):
        self.wrapper.setScansToAverage(self.chan,int(nAvg))

    def getScansToAverage(self):
        return self.wrapper.getScansToAverage(self.chan)

    def setCorrectForElectricalDark(self, nAvg):
        self.wrapper.setCorrectForElectricalDark(self.chan,int(nAvg))

    def getCorrectForElectricalDark(self):
        return self.wrapper.getCorrectForElectricalDark(self.chan)

    def setCorrectForDetectorNonlinearity(self, nAvg):
        self.wrapper.setCorrectForDetectorNonlinearity(self.chan,int(nAvg))

    def getCorrectForDetectorNonlinearity(self):
        return self.wrapper.getCorrectForDetectorNonlinearity(self.chan)
