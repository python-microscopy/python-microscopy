#!/usr/bin/python

###############
# PM100USB.py
#
# Copyright David Baddeley, 2012
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
################
import logging
import visa
logging.getLogger('pyvisa').setLevel(logging.INFO)

class PowerMeter:
    def __init__(self, ID='USB0::0x1313::0x8072::P2000343', dispScale=14):
        #self.instr = visa.instrument(ID)
        self._rm = visa.ResourceManager()
        self.resource = self._rm.open_resource(ID)
        self.instr = self.resource
        self.dispScale=dispScale

    def GetPower(self):
        return float(self.instr.ask('MEAS:POW?'))

    def GetWavelength(self):
        return float(self.instr.ask('SENS:CORR:WAV?'))

    def SetWavelength(self, wavelength):
        self.instr.write('SENS:CORR:WAV %d' % wavelength)

    def GetStatusText(self):
        try:
            pow = self.GetPower()
            return 'Laser power: %3.3f mW' % (pow*1e3*self.dispScale)
        except:
            return 'Laser power: ERR'


