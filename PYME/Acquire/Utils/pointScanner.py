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
import threading

import numpy as np
from PYME.Acquire import eventLog
import uuid
from PYME.contrib import dispatch
import logging
logger = logging.getLogger(__name__)

class PointScanner(object):
    def __init__(self, scope, pixels = 10, pixelsize=0.1, dwelltime = 1, background=0, avg=True, evtLog=False, sync=False,
                 trigger=False, stop_on_complete=False, return_to_start=True):
        """
        :param return_to_start: bool
            Flag to toggle returning home at the end of the scan. False leaves scope position as-is on scan completion.
        """
        self.scope = scope
        #self.xpiezo = xpiezo
        #self.ypiezo = ypiezo

        self.trigger = trigger

        self.dwellTime = dwelltime
        self.background = background
        self.avg = avg
        self.pixels = pixels
        self.pixelsize = pixelsize
        self._stop_on_complete = stop_on_complete
        self._return_to_start = return_to_start

        if np.isscalar(pixelsize):
            self.pixelsize = np.array([pixelsize, pixelsize])

        self.evtLog = evtLog
        self.sync = sync

        self._rlock = threading.Lock()
        
        self.running = False
        self._uuid = uuid.uuid4()
        self.on_stop = dispatch.Signal()

    def genCoords(self):
        self.currPos = self.scope.GetPos()
        logger.debug(self.currPos)
        
        if np.isscalar(self.pixels):
            #constant - use as number of pixels, center on current piezo position
            self.xp = self.pixelsize[0]*np.arange(-self.pixels/2, self.pixels/2 +1) + self.currPos['x']
            self.yp = self.pixelsize[1]*np.arange(-self.pixels/2, self.pixels/2 +1) + self.currPos['y']
        elif np.isscalar(self.pixels[0]):
            #a 1D array - numbers in either direction centered on piezo pos
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
        self.scope.frameWrangler.stop()
        self.scope.state.setItems({'Positioning.x' : self.xp[0], 'Positioning.y' : self.yp[0]}, stopCamera = True)
        if self.trigger:
            self.scope.cam.SetAcquisitionMode(self.scope.cam.MODE_SOFTWARE_TRIGGER)
        self.scope.frameWrangler.start()

        #if self.sync:
        #    while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
        #        time.sleep(.05)
        
        if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[0])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[0])


        #self.scope.frameWrangler.WantFrameNotification.append(self.tick)
        self.scope.frameWrangler.onFrame.connect(self.tick, dispatch_uid=self._uuid)

        if self.trigger:
            self.scope.cam.FireSoftwareTrigger()
        
        #if self.sync:
        #    self.scope.frameWrangler.HardwareChecks.append(self.onTarget)

    def onTarget(self):
        #FIXME
        return self.xpiezo[0].onTarget
    
    def _position_for_index(self, callN):
        # todo - precalculate ???
        x_i = callN % self.nx
        y_i = int((callN % (self.imsize)) / self.nx)
    
        # do a bidirectional scan(faster)
        if ((y_i) % 2):
            #scan in reverse direction on odd runs
            new_x = self.xp[(len(self.xp) - 1) - x_i]
        else:
            new_x = self.xp[x_i]
            
        new_y = self.yp[y_i]
        
        return new_x, new_y

    def tick(self, frameData, **kwargs):
        with self._rlock:
            if not self.running:
                return

            try:
                cam_trigger = self.scope.cam.GetAcquisitionMode() == self.scope.cam.MODE_SOFTWARE_TRIGGER
            except AttributeError:
                cam_trigger = False

            #logger.debug('Cam_trigger: %s' % repr(cam_trigger))

            #print self.callNum
            if (self.callNum % self.dwellTime) == 0:
                #record pixel in overview
                callN = int(self.callNum/self.dwellTime)
                if self.avg:
                    self.image[callN % self.nx, int((callN % (self.image.size))/self.nx)] = self.scope.currentFrame.mean() - self.background
                    self.view.Refresh()
            
            if self.callNum >= self.dwellTime * self.imsize - 1:
                # we've acquired the last frame
                if self._stop_on_complete:
                    self._stop()
                    return

            if ((self.callNum +1) % self.dwellTime) == 0:
                #move piezo
                callN = int((self.callNum+1)/self.dwellTime)
                new_x, new_y = self._position_for_index(callN)

                self.scope.state.setItems({'Positioning.x' : new_x,
                                           'Positioning.y' : new_y
                                           }, stopCamera = not cam_trigger)

                #print 'SetP'

                if self.evtLog:
                    #eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[callN % self.nx])
                    #eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[(callN % (self.imsize))/self.nx])
                    eventLog.logEvent('ScannerXPos', '%3.6f' % self.scope.state['Positioning.x'])
                    eventLog.logEvent('ScannerYPos', '%3.6f' % self.scope.state['Positioning.y'])

            if cam_trigger:
                #logger.debug('Firing camera trigger')
                self.scope.cam.FireSoftwareTrigger()
                if self.evtLog:
                    eventLog.logEvent('StartAq',"")
#
        #
        #if self.sync:
#                while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
#                    time.sleep(.05)

        self.callNum += 1
        
    def _stop(self, send_stop=True):
        self.running = False
        #self.xpiezo[0].MoveTo(self.xpiezo[1], self.currPos[0])
        #self.ypiezo[0].MoveTo(self.ypiezo[1], self.currPos[1])
    
        #self.scope.SetPos(**self.currPos)
        try:
            #self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
            self.scope.frameWrangler.onFrame.disconnect(self.tick, dispatch_uid=self._uuid)
            #if self.sync:
            #    self.scope.frameWrangler.HardwareChecks.remove(self.onTarget)
        except:
            logger.exception('Could not disconnect pointScanner tick from frameWrangler.onFrame')
    
        self.scope.frameWrangler.stop()

        if send_stop:
            # optionally defer sending the stop signal until after a derived class _stop
            # method has run
            self.on_stop.send(self)
        
        if self._return_to_start:
            logger.debug('Returning home : %s' % self.currPos)
            self.scope.state.setItems({'Positioning.x': self.currPos['x'],
                                       'Positioning.y': self.currPos['y'],
                                       }, stopCamera=True)
        
        self.scope.turnAllLasersOff()
    
        if self.trigger:
            self.scope.cam.SetAcquisitionMode(self.scope.cam.MODE_CONTINUOUS)
    
        self.scope.frameWrangler.start()

    #def __del__(self):
    #    self.scope.frameWrangler.WantFrameNotification.remove(self.tick)
    def stop(self):
        with self._rlock:
            self._stop()
            

class CircularPointScanner(PointScanner):
    def genCoords(self):
        """
        Generate coordinates for square ROIs evenly distributed within a circle. Order them first by radius, and then
        by increasing theta such that the initial position is scanned first, and then subsequent points are scanned in
        an ~optimal order.
        """
        self.currPos = self.scope.GetPos()
        logger.debug('Current positions: %s' % (self.currPos,))
    
        r, t = [0], [np.array([0])]
        for r_ring in self.pixelsize[0] * np.arange(1, self.pixels + 1):  # 0th ring is (0, 0)
            # keep the rings spaced by pixel size and hope the overlap is enough
            # 2 pi / (2 pi r / pixsize) = pixsize/r
            thetas = np.arange(0, 2 * np.pi, self.pixelsize[0] / r_ring)
            r.extend(r_ring * np.ones_like(thetas))
            t.append(thetas)
    
        # convert to cartesian and add currPos offset
        r = np.asarray(r)
        t = np.concatenate(t)
        self.xp = r * np.cos(t) + self.currPos['x']
        self.yp = r * np.sin(t) + self.currPos['y']
    
        self.nx = len(self.xp)
        self.ny = len(self.yp)
        self.imsize = self.nx

    def _position_for_index(self, callN):
        ind = callN % self.nx
        return self.xp[ind], self.yp[ind]


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
            callN = int(self.callNum/self.dwellTime)

            #vary z fastest
            iz = callN % self.nz
            ix = (callN/self.nz) % self.nx
            iy = (callN/(self.nz*self.nx)) % self.ny

            if self.avg:
                self.image[ix, iy, iz] = self.ds.mean() - self.background
                self.view.Refresh()

        if ((self.callNum +1) % self.dwellTime) == 0:
            #move piezo
            callN = int((self.callNum+1)/self.dwellTime)

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

