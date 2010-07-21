from PYME.DSView.dsviewer_npy import View3D
from PYME import cSMI
import numpy as np
from PYME.Acquire import eventLog

class PointScanner:
    def __init__(self, xpiezo, ypiezo, scope, pixels = 10, pixelsize=0.1, dwelltime = 1, background=0, avg=True, evtLog=False):
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

        self.scope.pa.WantFrameNotification.append(self.tick)

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

        self.callNum += 1

    #def __del__(self):
    #    self.scope.pa.WantFrameNotification.remove(self.tick)
    def stop(self):
        self.scope.pa.WantFrameNotification.remove(self.tick)



