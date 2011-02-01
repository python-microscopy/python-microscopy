import time
from PYME.DSView.dsviewer_npy import View3D
from PYME import cSMI
import numpy as np
from PYME.Acquire import eventLog

class PointScanner:
    def __init__(self, xpiezo, ypiezo, scope, pixels = 10, pixelsize=0.1, dwelltime = 1, background=0, avg=True, evtLog=False, sync=False):
        self.scope = scope
        self.xpiezo = xpiezo
        self.ypiezo = ypiezo

        self.dwellTime = dwelltime
        self.background = background
        self.avg = avg
        self.pixels = pixels
        self.pixelsize = pixelsize

        if np.isscalar(pixelsize):
            self.pixelsize = np.array([pixelsize, pixelsize])

        self.evtLog = evtLog
        self.sync = sync

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

        self.nx = len(self.xp)
        self.ny = len(self.yp)

        self.currPos = (self.xpiezo[0].GetPos(self.xpiezo[1]), self.ypiezo[0].GetPos(self.ypiezo[1]))

        self.imsize = self.nx*self.ny

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

        self.callNum = 0

        if self.avg:
            self.image = np.zeros((self.nx, self.ny))

            self.ds = cSMI.CDataStack_AsArray(scope.pa.ds, 0)

            self.view = View3D(self.image)

        self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[0])
        self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[0])

        #if self.sync:
        #    while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
        #        time.sleep(.05)
        
        if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[0])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[0])


        self.scope.pa.WantFrameNotification.append(self.tick)
        
        if self.sync:
            self.scope.pa.HardwareChecks.append(self.onTarget)

    def onTarget(self):
        return self.xpiezo[0].onTarget

    def tick(self, caller=None):
        #print self.callNum
        if (self.callNum % self.dwellTime) == 0:
            #record pixel in overview
            callN = self.callNum/self.dwellTime
            if self.avg:
                self.image[callN % self.nx, (callN % (self.image.size))/self.nx] = self.ds.mean() - self.background
                self.view.Refresh()

        if ((self.callNum +1) % self.dwellTime) == 0:
            #move piezo
            callN = (self.callNum+1)/self.dwellTime
            self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[callN % self.nx])
            self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[(callN % (self.imsize))/self.nx])
            if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[callN % self.nx])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[(callN % (self.imsize))/self.nx])

#            if self.sync:
#                while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
#                    time.sleep(.05)

        self.callNum += 1

    #def __del__(self):
    #    self.scope.pa.WantFrameNotification.remove(self.tick)
    def stop(self):
        self.xpiezo[0].MoveTo(self.xpiezo[1], self.currPos[0])
        self.ypiezo[0].MoveTo(self.ypiezo[1], self.currPos[1])
        
        try:
            self.scope.pa.WantFrameNotification.remove(self.tick)
            if self.sync:
                self.scope.pa.HardwareChecks.remove(self.onTarget)
        finally:
            pass



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
        self.zp = np.arange(self.scope.sa.GetStartPos(), self.scope.sa.GetEndPos()+.95*self.scope.sa.GetStepSize(),self.scope.sa.GetStepSize())

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

        self.callNum = 0

        if self.avg:
            self.image = np.zeros((self.nx, self.ny, self.nz))

            self.ds = cSMI.CDataStack_AsArray(scope.pa.ds, 0)

            self.view = View3D(self.image)

        self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[0])
        self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[0])
        self.zpiezo[0].MoveTo(self.zpiezo[1], self.zp[0])

        #if self.sync:
        #    while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
        #        time.sleep(.05)

        if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[0])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[0])
                eventLog.logEvent('ScannerZPos', '%3.6f' % self.zp[0])


        self.scope.pa.WantFrameNotification.append(self.tick)

        if self.sync:
            self.scope.pa.HardwareChecks.append(self.onTarget)

    def onTarget(self):
        return self.xpiezo[0].onTarget

    def tick(self, caller=None):
        #print self.callNum
        
        if (self.callNum % self.dwellTime) == 0:
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
            self.xpiezo[0].MoveTo(self.xpiezo[1], self.xp[ix])
            self.ypiezo[0].MoveTo(self.ypiezo[1], self.yp[iy])
            if self.evtLog:
                eventLog.logEvent('ScannerXPos', '%3.6f' % self.xp[ix])
                eventLog.logEvent('ScannerYPos', '%3.6f' % self.yp[iy])
                eventLog.logEvent('ScannerZPos', '%3.6f' % self.zp[iz])

#            if self.sync:
#                while not self.xpiezo[0].IsOnTarget(): #wait for stage to move
#                    time.sleep(.05)

        self.callNum += 1

    #def __del__(self):
    #    self.scope.pa.WantFrameNotification.remove(self.tick)
    def stop(self):
        self.xpiezo[0].MoveTo(self.xpiezo[1], self.currPos[0])
        self.ypiezo[0].MoveTo(self.ypiezo[1], self.currPos[1])
        self.zpiezo[0].MoveTo(self.zpiezo[1], self.currPos[2])

        try:
            self.scope.pa.WantFrameNotification.remove(self.tick)
            if self.sync:
                self.scope.pa.HardwareChecks.remove(self.onTarget)
        finally:
            pass
        




