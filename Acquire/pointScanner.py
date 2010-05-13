from PYME.DSView.dsviewer_npy import View3D
from PYME import cSMI
import numpy as np
from PYME.Acquire import eventLog

class PointScanner:
    def __init__(self, xpiezo, ypiezo, scope, pixels = 10, pixelSize=0.1, dwelltime = 1, background=0):
        self.scope = scope
        self.xpiezo = xpiezo
        self.ypiezo = ypiezo
        self.dwellTime = dwelltime
        self.background = background
        #pixels = np.array(pixels)

        if np.isscalar(pixels):
            #constant - use as number of pixels
            #center on current piezo position
            self.xp = pixelSize*np.arange(-pixels/2, pixels/2 +1) + xpiezo.GetPos()
            self.yp = pixelSize*np.arange(-pixels/2, pixels/2 +1) + ypiezo.GetPos()
        elif np.isscalar(pixels[0]):
            #a 1D array - numbers in either direction centered on piezo pos
            self.xp = pixelSize*np.arange(-pixels[0]/2, pixels[0]/2 +1) + xpiezo.GetPos()
            self.yp = pixelSize*np.arange(-pixels[1]/2, pixels[1]/2 +1) + ypiezo.GetPos()
        else:
            #actual pixel positions
            self.xp = pixels[0]
            self.yp = pixels[1]

        self.nx = len(self.xp)
        self.ny = len(self.yp)

        self.image = np.zeros((self.nx, self.ny))


        self.callNum = 0
        self.ds = cSMI.CDataStack_AsArray(scope.pa.ds, 0)

        self.view = View3D(self.image)

        self.xpiezo.MoveTo(0, self.xp[0])
        self.ypiezo.MoveTo(0, self.yp[0])

        self.scope.pa.WantFrameNotification.append(self.tick)

    def tick(self, caller=None):
        #print self.callNum
        if (self.callNum % self.dwellTime) == 0:
            #record pixel in overview
            callN = self.callNum/self.dwellTime
            self.image[callN % self.nx, (callN % (self.image.size))/self.nx] = self.ds.mean() - self.background
            self.view.Refresh()

        if ((self.callNum +1) % self.dwellTime) == 0:
            #move piezo
            callN = (self.callNum+1)/self.dwellTime
            self.xpiezo.MoveTo(0, self.xp[callN % self.nx])
            self.ypiezo.MoveTo(0, self.yp[(callN % (self.image.size))/self.nx])
            eventLog.logEvent('ScannerXPos', '%3.3f' % self.xp[callN % self.nx])
            eventLog.logEvent('ScannerYPos', '%3.3f' % self.yp[(callN % (self.image.size))/self.nx])

        self.callNum += 1

    def __del__(self):
        self.scope.pa.WantFrameNotification.remove(self.tick)



