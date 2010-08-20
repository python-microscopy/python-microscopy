import numpy as np
from PYME.DSView.dsviewer_npy import View3D

PIXELSIZE = (8*1.5*.07)/1e3

class fastTiler:
    def __init__(self, scope, ystep=8, pixelsize=PIXELSIZE):
        self.scope = scope
        self.i = -1
        self.j = 0
        self.dir = 1
        self.ystep = ystep
        self.pixelsize = pixelsize
        self.runInProgress = True

        self.rect = self.GetBoundingRect()
        self.data = np.zeros((self.rect[2]/pixelsize + 64, self.rect[3]/pixelsize + 64), 'uint16')

        #self.startPositions = []
        #self.endYPositions = []
        self.GenRunPositions()

        View3D(self.data, title='tiled image')
        #self.GotoStart()
        self.scope.pa.WantFrameNotification.append(self.OnTick)
        #self.scope.pa.WantFrameGroupNotification.append(self)


    def OnTick(self, caller=None):
        if self.i >=0 and self.i < (self.data.shape[1]+64) and self.scope.stage.IsMoving(1):
            self.data[self.j:(self.j+64), self.i:(self.i + 63)] = self.scope.pa.dsa[:,1:,0][:,::-1]
            self.i += self.dir*self.ystep
        elif self.scope.stage.IsMoving(0) or self.scope.stage.IsMoving(1):
            # positioning for the start of a run - do nothing
            pass
        elif self.runInProgress: #we've got to the end of our run - position for the next
            self.runInProgress=False
            print 'foo'
            if len(self.startPositions) > 0:
                nextX, nextY = self.startPositions.pop(0)
                self.scope.stage.MoveTo(0, nextX)
                self.scope.stage.MoveTo(1, nextY)
            else: #gone through all start positions -> we're done
                self.scope.pa.WantFrameNotification.remove(self.OnTick)
                #View3D(self.data, title='tiled image')

        else: #we've got to the next starting position - fire off next run
            xp = self.scope.stage.GetPos(0)
            yp = self.scope.stage.GetPos(1)
            
            self.j = int((xp - self.rect[0])/self.pixelsize)
            self.i = int((yp - self.rect[1])/self.pixelsize)

            nextY = self.endYPositions.pop(0)

            if nextY > yp: #going forward
                self.dir = 1
            else: #going backwards
                self.dir = -1

            self.scope.stage.MoveTo(1, nextY)
            self.runInProgress = True



class fastRectTiler(fastTiler):
    def __init__(self, scope, rect, xstep = .00005, ystep=8, pixelsize=PIXELSIZE):
        self.rect = rect
        self.xstep = xstep
        self.pixelsize = pixelsize

        fastTiler.__init__(self, scope, ystep=ystep, pixelsize=pixelsize)
        print self.startPositions

    def GetBoundingRect(self):
        return self.rect

    def GenRunPositions(self):
        xposs = self.rect[0] + np.arange(0, self.rect[2], max(self.xstep, 64*self.pixelsize))
        yposs = [self.rect[1], self.rect[1] + self.rect[3]]

        self.startPositions = []
        self.endYPositions = []

        for i in range(len(xposs)):
            self.startPositions.append((xposs[i], yposs[i%2]))
            self.endYPositions.append(yposs[(i+1)%2])







