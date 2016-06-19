#!/usr/bin/python

###############
# pointScanner.py
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
#import time
#from PYME.DSView.dsviewer import View3D

import numpy as np
from PYME.Acquire import eventLog

class PointScanner:
    def __init__(self, scope, LC, evtLog=False, sync=False, stepSize = 3, pointSize=3, pointSpacing = 100):
        self.scope = scope
        self.LC = LC
        
        self.stepSize = stepSize
        self.pointSize = pointSize
        self.pointSpacing = pointSpacing

        self.evtLog = evtLog
        self.sync = sync
        
        self.x = 0
        self.y = 0

    def SetPattern(self, ph1, ph2):
        m = (((self.LC.X + ph1) % self.pointSpacing) <= self.pointSize)*(((self.LC.Y + ph2) % self.pointSpacing) <= self.pointSize)
        self.LC.SetMask(m)    

    def start(self):
        self.x = 0
        self.y = 0
        
        self.SetPattern(self.x, self.y)

        self.scope.frameWrangler.WantFrameNotification.append(self.tick)
        
        
    def tick(self, caller=None):
        self.y += self.stepSize
        if self.y > self.pointSpacing:
            self.y = 0
            self.x += self.stepSize
            if self.x > self.pointSpacing:
                self.x = 0
        
        self.SetPattern(self.x, self.y)
        

    #def __del__(self):
    #    self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
    def stop(self):
        
        
        try:
            self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
            
        finally:
            pass



