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
import time
from PYME.DSView.dsviewer import View3D

import numpy as np
from PYME.Acquire import eventLog

class PointScanner:
    def __init__(self, scope, pixels = 10, pixelsize=0.1, dwelltime = 1, background=0, avg=True, evtLog=False, sync=False):
        self.scope = scope
        #self.xpiezo = xpiezo
        #self.ypiezo = ypiezo

        self.dwellTime = dwelltime
        self.background = background
        self.avg = avg
        self.pixels = pixels
        self.pixelsize = pixelsize

        if np.isscalar(pixelsize):
            self.pixelsize = np.array([pixelsize, pixelsize])

        self.evtLog = evtLog
        self.sync = sync
        
        self.running = False

    def genCoords(self):
        self.currPos = self.scope.GetPos()
        
        if np.isscalar(self.pixels):
            #constant - use as number of pixels
            #center on current piezo position
            #print self.pixelsize[0]
            self.xp = self.pixelsize[0]*np.arange(-self.pixels/2, self.pixels/2 +1) + self.currPos['x']
            self.yp = self.pixelsize[1]*np.arange(-self.pixels/2, self.pixels/2 +1) + self.currPos['y']
        elif np.isscalar(self.pixels[0]):
            #a 1D array - numbers in either direction centered on piezo pos
            #print self.pixelsize[0]
            self.xp = self.pixelsize[0]*np.arange(-self.pixels[0]/2, self.pixels[0]/2 +1) + self.currPos['x']
            self.yp = self.pixelsize[1]*np.arange(-self.pixels[1]/2, self.pixels[1]/2 +1) + self.currPos['y']
        else:
            #actual pixel positions
            self.xp = self.pixels[0]
            self.yp = self.pixels[1]

        self.nx = len(self.xp)
        self.ny = len(self.yp)

        #self.currPos = (self.xpiezo[0].GetPos(self.xpiezo[1]), self.ypiezo[0].GetPos(self.ypiezo[1]))

        self.imsize = self.nx*self.ny
        

    def start(self):
        self.running = True
        
        #pixels = np.array(pixels)

#        if np.isscalar(self.pixels):
#            #constant - use as number of pixels
#            #center on current piezo position
#            self.xp = self.pixelsize*np.arange(-self.pixels/2, self.pixels/2 +1) + self.xpiezo[0].GetPos(self.xpiezo[1])
#            self.yp = self.pixelsize*np.arange(-self.pixels/2, self.pixels/2 +1) + self.ypiezo[0].GetPos(self.ypiezo[1])
#        elif np.isscalar(self.pixels[0]):
#            #a 1D array - numbers in either direction centered on piezo pos
#            self.xp = self.pixelsize*np.arange(-self.pixels[0]/2, self.pixels[0]/2 +1) + self.xpiezo[0].GetPos(self.xpiezo[1])
#            self.yp = self.pixelsize*np.arange(-self.pixels[1]/2, self.pixels[1]/2 +1) + self.ypiezo[0].GetPos(self.ypiezo[1])
#        else:
#            #actual pixel positions
#            self.xp = self.pixels[0]
#            self.yp = self.pixels[1]
#
#        self.nx = len(self.xp)
#        self.ny = len(self.yp)
#
#        self.imsize = self.nx*self.ny

        self.genCoords()

        self.callNum = 0

        if self.avg:
            self.image = np.zeros((self.nx, self.ny))

            #self.ds = scope.frameWrangler.currentFrame

            self.view = View3D(self.image)

        #self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[0])
        #self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[0])

        #self.scope.SetPos(x=self.xp[0], y = self.yp[0])
        self.scope.state.setItems({'Positioning.x' : self.xp[0], 'Positioning.y' : self.yp[0]}, stopCamera = True)

        #if self.sync:
        #    while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
        #        time.sleep(.05)
        
        if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[0])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[0])


        #self.scope.frameWrangler.WantFrameNotification.append(self.tick)
        self.scope.frameWrangler.onFrame.connect(self.tick)
        
        #if self.sync:
        #    self.scope.frameWrangler.HardwareChecks.append(self.onTarget)

    def onTarget(self):
        return self.xpiezo[0].onTarget

    def tick(self, frameData, **kwargs):
        if not self.running:
            return
        #print self.callNum
        if (self.callNum % self.dwellTime) == 0:
            #record pixel in overview
            callN = self.callNum/self.dwellTime
            if self.avg:
                self.image[callN % self.nx, (callN % (self.image.size))/self.nx] = self.scope.currentFrame.mean() - self.background
                self.view.Refresh()

        if ((self.callNum +1) % self.dwellTime) == 0:
            #move piezo
            callN = (self.callNum+1)/self.dwellTime
            
            #self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[callN % self.nx])
            #self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[(callN % (self.imsize))/self.nx])
            
            #self.scope.SetPos(x=self.xp[callN % self.nx], y = self.yp[(callN % (self.imsize))/self.nx])
            self.scope.state.setItems({'Positioning.x' : self.xp[callN % self.nx], 
                                       'Positioning.y' : self.yp[(callN % (self.imsize))/self.nx]
                                       }, stopCamera = True)
                                       
            #print 'SetP'
            
            if self.evtLog:
                #eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[callN % self.nx])
                #eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[(callN % (self.imsize))/self.nx])
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.scope.state['Positioning.x'])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.scope.state['Positioning.y'])

#            if self.sync:
#                while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
#                    time.sleep(.05)

        self.callNum += 1

    #def __del__(self):
    #    self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
    def stop(self):
        self.running = False
        #self.xpiezo[0].MoveTo(self.xpiezo[1], self.currPos[0])
        #self.ypiezo[0].MoveTo(self.ypiezo[1], self.currPos[1])
    
        #self.scope.SetPos(**self.currPos)
        try:
            #self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
            self.scope.frameWrangler.onFrame.disconnect(self.tick)
            #if self.sync:
            #    self.scope.frameWrangler.HardwareChecks.remove(self.onTarget)
        finally:
            pass
    
        print('Returning home : %s' % self.currPos)
        
        self.scope.state.setItems({'Positioning.x' : self.currPos['x'], 
                                       'Positioning.y' : self.currPos['y'],
                                       }, stopCamera = True)
                                       
        
        



class PointScanner3D:
    def __init__(self, xpiezo, ypiezo, zpiezo, scope, pixels = 10, pixelsize=0.1, dwelltime = 1, background=0, avg=True, evtLog=False, sync=True):
        self.scope = scope
        self.xpiezo = xpiezo
        self.ypiezo = ypiezo
        self.zpiezo = zpiezo

        self.dwellTime = dwelltime
        self.background = background
        self.avg = avg
        self.pixels = pixels
        self.pixelsize = pixelsize

        if np.isscalar(pixelsize):
            self.pixelsize = np.array([pixelsize, pixelsize])

        self.evtLog = evtLog
        self.sync = sync

        self.ix_o = -1
        self.iy_o = -1

    def genCoords(self):
        if np.isscalar(self.pixels):
            #constant - use as number of pixels
            #center on current piezo position
            #print self.pixelsize[0]
            self.xp = self.pixelsize[0]*np.arange(-self.pixels/2, self.pixels/2 +1) + self.xpiezo[0].GetPos(self.xpiezo[1])
            self.yp = self.pixelsize[1]*np.arange(-self.pixels/2, self.pixels/2 +1) + self.ypiezo[0].GetPos(self.ypiezo[1])
        elif np.isscalar(self.pixels[0]):
            #a 1D array - numbers in either direction centered on piezo pos
            #print self.pixelsize[0]
            self.xp = self.pixelsize[0]*np.arange(-self.pixels[0]/2, self.pixels[0]/2 +1) + self.xpiezo[0].GetPos(self.xpiezo[1])
            self.yp = self.pixelsize[1]*np.arange(-self.pixels[1]/2, self.pixels[1]/2 +1) + self.ypiezo[0].GetPos(self.ypiezo[1])
        else:
            #actual pixel positions
            self.xp = self.pixels[0]
            self.yp = self.pixels[1]

        #get z range from normal z controls
        self.zp = np.arange(self.scope.stackSettings.GetStartPos(), self.scope.stackSettings.GetEndPos()+.95*self.scope.stackSettings.GetStepSize(),self.scope.stackSettings.GetStepSize())

        self.nx = len(self.xp)
        self.ny = len(self.yp)
        self.nz = len(self.zp)

        self.currPos = (self.xpiezo[0].GetPos(self.xpiezo[1]), self.ypiezo[0].GetPos(self.ypiezo[1]), self.zpiezo[0].GetPos(self.zpiezo[1]))

        self.imsize = self.nx*self.ny*self.nz

    def start(self):

        #pixels = np.array(pixels)

#        if np.isscalar(self.pixels):
#            #constant - use as number of pixels
#            #center on current piezo position
#            self.xp = self.pixelsize*np.arange(-self.pixels/2, self.pixels/2 +1) + self.xpiezo[0].GetPos(self.xpiezo[1])
#            self.yp = self.pixelsize*np.arange(-self.pixels/2, self.pixels/2 +1) + self.ypiezo[0].GetPos(self.ypiezo[1])
#        elif np.isscalar(self.pixels[0]):
#            #a 1D array - numbers in either direction centered on piezo pos
#            self.xp = self.pixelsize*np.arange(-self.pixels[0]/2, self.pixels[0]/2 +1) + self.xpiezo[0].GetPos(self.xpiezo[1])
#            self.yp = self.pixelsize*np.arange(-self.pixels[1]/2, self.pixels[1]/2 +1) + self.ypiezo[0].GetPos(self.ypiezo[1])
#        else:
#            #actual pixel positions
#            self.xp = self.pixels[0]
#            self.yp = self.pixels[1]
#
#        self.nx = len(self.xp)
#        self.ny = len(self.yp)
#
#        self.imsize = self.nx*self.ny

        self.genCoords()

        self.callNum = -1

        if self.avg:
            self.image = np.zeros((self.nx, self.ny, self.nz))

            self.ds = self.scope.frameWrangler.currentFrame

            self.view = View3D(self.image)

        self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[0])
        self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[0])
        self.zpiezo[0].MoveTo(self.zpiezo[1], self.zp[0])

        self.ix_o = 0
        self.iy_o = 0

        #if self.sync:
        #    while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
        #        time.sleep(.05)

        if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[0])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[0])
                eventLog.logEvent('ScannerZPos', '%3.6f' % self.zp[0])


        self.scope.frameWrangler.WantFrameNotification.append(self.tick)

        if self.sync:
            self.scope.frameWrangler.HardwareChecks.append(self.onTarget)

    def onTarget(self):
        return self.xpiezo[0].onTarget

    def tick(self, caller=None):
        #print self.callNum
        
        if self.callNum > 0  and (self.callNum % self.dwellTime) == 0:
            #record pixel in overview
            callN = self.callNum/self.dwellTime

            #vary z fastest
            iz = callN % self.nz
            ix = (callN/self.nz) % self.nx
            iy = (callN/(self.nz*self.nx)) % self.ny

            if self.avg:
                self.image[ix, iy, iz] = self.ds.mean() - self.background
                self.view.Refresh()

        if ((self.callNum +1) % self.dwellTime) == 0:
            #move piezo
            callN = (self.callNum+1)/self.dwellTime

            #vary z fastest
            iz = callN % self.nz
            ix = (callN/self.nz) % self.nx
            iy = (callN/(self.nz*self.nx)) % self.ny

            self.zpiezo[0].MoveTo(self.zpiezo[1], self.zp[iz])
            if self.evtLog:
                eventLog.logEvent('ScannerZPos', '%3.6f' % self.zp[iz])

            if not ix == self.ix_o:
                self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[ix])
                self.ix_o = ix
                if self.evtLog:
                    eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[ix])
                    
            if not iy == self.iy_o:
                self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[iy])
                self.iy_o = iy
                if self.evtLog:
                    eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[iy])

                        

#            if self.sync:
#                while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
#                    time.sleep(.05)

        self.callNum += 1

    #def __del__(self):
    #    self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
    def stop(self):
        self.xpiezo[0].MoveTo(self.xpiezo[1], self.currPos[0])
        self.ypiezo[0].MoveTo(self.ypiezo[1], self.currPos[1])
        self.zpiezo[0].MoveTo(self.zpiezo[1], self.currPos[2])

        try:
            self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
            if self.sync:
                self.scope.frameWrangler.HardwareChecks.remove(self.onTarget)
        finally:
            pass
        




