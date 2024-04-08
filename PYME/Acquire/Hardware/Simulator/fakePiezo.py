#!/usr/bin/python

##################
# fakePiezo.py
#
# Copyright David Baddeley, 2009
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

from PYME.Acquire.Hardware.Piezos.base_piezo import PiezoBase
import threading
import numpy as np
import time

class FakePiezo(PiezoBase):
    gui_description = 'Fake %s-piezo'
    units_um = 1.0
    
    def __init__(self, maxtravel = 100.00):
        self.max_travel = maxtravel
        self.curpos = maxtravel/2.0
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True, vel=1.0):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.curpos=fPos
            else:
                self.curpos=self.max_travel
        else:
            self.curpos=0

    def MoveRel(self, iChannel, incr, bTimeOut=True):
        self.MoveTo(iChannel, self.curpos+incr)

    def GetPos(self, iChannel=1):
        return self.curpos

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 1

    def GetMin(self,iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel

    def __getattr__(self, name):
        if name == 'lastPos':
            return self.curpos
        else: 
            raise AttributeError(name)  # <<< DON'T FORGET THIS LINE !!
        
class DriftyFakePiezo(FakePiezo):
    """Fake piezo that drifts over time using a random-walk model. Drift rate is specified in um/s^2.
    
    We use a 100ms time step.
    """
    def __init__(self, maxtravel = 100.00, drift_rate = 0.01):
        FakePiezo.__init__(self, maxtravel)
        self._100ms_drift_rate = drift_rate*np.sqrt(0.1)

        self.drift = 0.0 # initial drift

        self._drift_thread = threading.Thread(target=self._drift)
        self._drift_thread.setDaemon(True)
        self._drift_thread.start()

    def _drift(self):
        while True:
            self.drift += self._100ms_drift_rate*np.random.randn()
            time.sleep(0.1)

    @property
    def effective_pos(self):
        return self.curpos + self.drift
