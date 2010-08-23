import numpy as np
from PYME.DSView.dsviewer_npy import View3D
from math import floor

PIXELSIZE = (16*1.5*.072)/1e3

class fastTiler:
    def __init__(self, scope, ystep=10.5, pixelsize=PIXELSIZE):
        self.scope = scope
        self.i = -1
        self.j = 0
        self.dir = 1
        self.ystep = ystep
        self.yspeed = ystep
        self.pixelsize = pixelsize
        self.runInProgress = True

        self.rect = self.GetBoundingRect()
        self.data = np.zeros((self.rect[2]/pixelsize + 32, self.rect[3]/pixelsize + 33), 'uint16')

        #self.startPositions = []
        #self.endYPositions = []
        self.GenRunPositions()

        self.visfr = View3D(self.data, title='tiled image')
        #self.GotoStart()
        self.scope.pa.WantFrameNotification.append(self.OnTick)
        self.scope.pa.WantFrameGroupNotification.append(self.updateView)

    def updateView(self, caller=None):
        self.visfr.vp.Refresh()

    def detach(self):
        self.scope.pa.WantFrameNotification.remove(self.OnTick)
        self.scope.pa.WantFrameGroupNotification.remove(self.updateView)


    def OnTick(self, caller=None):
        #if self.scope.stage.moving[0]:
        #    print self.i
        if self.runInProgress and self.i >=0 and self.i < (self.data.shape[1]-32):# and self.scope.stage.moving[1] and not self.scope.stage.moving[0]:
            #print self.i, self.j
            self.data[self.j:(self.j+32), floor(self.i):(floor(self.i) + 15)] = np.maximum(self.scope.pa.dsa[:,1:16,0] - (self.scope.cam.ADOffset), 0)
            self.i += self.dir*self.yspeed
            if self.yspeed < self.ystep:
                self.yspeed += 1
        elif self.scope.stage.moving[0] or self.scope.stage.moving[1]:
            # positioning for the start of a run - do nothing
            pass
        elif self.runInProgress: #we've got to the end of our run - position for the next
            self.scope.pa.stop()
            self.runInProgress=False
            self.i = -1
            print 'foo'
            if len(self.startPositions) > 0:
                nextX, nextY = self.startPositions.pop(0)
                self.scope.stage.MoveTo(0, nextX)
                self.scope.stage.MoveTo(1, nextY)
                self.scope.stage.moving = [1,1]
            else: #gone through all start positions -> we're done
                self.scope.pa.WantFrameNotification.remove(self.OnTick)
                self.scope.pa.WantFrameGroupNotification.remove(self.updateView)
                #View3D(self.data, title='tiled image')
            self.scope.pa.start()

        else: #we've got to the next starting position - fire off next run
            xp = self.scope.stage.GetPos(0)
            yp = self.scope.stage.GetPos(1)
            
            self.j = int((xp - self.rect[0])/self.pixelsize)
            self.i = int((yp - self.rect[1])/self.pixelsize)

            print self.i, self.j, self.data.shape

            nextY = self.endYPositions.pop(0)

            if nextY > yp: #going forward
                self.dir = 1
            else: #going backwards
                self.dir = -1

            self.yspeed = self.ystep

            self.scope.pa.stop()
            #self.scope.pa.start()
            self.scope.pa.purge()
            self.scope.stage.MoveTo(1, nextY)
            self.runInProgress = True
            self.scope.stage.moving = [1,1]
            #self.scope.pa.stop()
            self.scope.pa.start()




class fastRectTiler(fastTiler):
    def __init__(self, scope, rect, xstep = .00005, ystep=10.5, pixelsize=PIXELSIZE):
        self.rect = rect
        self.xstep = xstep
        self.pixelsize = pixelsize

        fastTiler.__init__(self, scope, ystep=ystep, pixelsize=pixelsize)
        print self.startPositions

    def GetBoundingRect(self):
        return self.rect

    def GenRunPositions(self):
        xposs = self.rect[0] + np.arange(0, self.rect[2], max(self.xstep, 32*self.pixelsize))
        yposs = [self.rect[1], self.rect[1] + self.rect[3]]

        self.startPositions = []
        self.endYPositions = []

        for i in range(len(xposs)):
            self.startPositions.append((xposs[i], yposs[i%2]))
            self.endYPositions.append(yposs[(i+1)%2])







