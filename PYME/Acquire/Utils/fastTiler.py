#!/usr/bin/python

###############
# fastTiler.py
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
import numpy as np
from PYME.DSView.dsviewer import View3D
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
        #self.scope.frameWrangler.WantFrameNotification.append(self.OnTick)
        #self.scope.frameWrangler.WantFrameGroupNotification.append(self.updateView)
        
        self.scope.frameWrangler.onFrame.connect(self.OnTick)
        self.scope.frameWrangler.onFrameGroup.connect(self.updateView)

    def updateView(self, **kwargs):
        self.visfr.vp.Refresh()

    def detach(self):
        #self.scope.frameWrangler.WantFrameNotification.remove(self.OnTick)
        #self.scope.frameWrangler.WantFrameGroupNotification.remove(self.updateView)
        self.scope.frameWrangler.onFrame.disconnect(self.OnTick)
        self.scope.frameWrangler.onFrameGroup.disconnect(self.updateView)


    def OnTick(self, sender, frameData, **kwargs):
        #if self.scope.stage.moving[0]:
        #    print self.i
        if self.runInProgress and self.i >=0 and self.i < (self.data.shape[1]-32):# and self.scope.stage.moving[1] and not self.scope.stage.moving[0]:
            #print self.i, self.j
            self.data[self.j:(self.j+32), floor(self.i):(floor(self.i) + 15)] = np.maximum(frameData[:,1:16,0] - (self.scope.cam.ADOffset), 0)
            self.i += self.dir*self.yspeed
            if self.yspeed < self.ystep:
                self.yspeed += 1
        elif self.scope.stage.moving[0] or self.scope.stage.moving[1]:
            # positioning for the start of a run - do nothing
            pass
        elif self.runInProgress: #we've got to the end of our run - position for the next
            self.scope.frameWrangler.stop()
            self.runInProgress=False
            self.i = -1
            #print('foo')
            if len(self.startPositions) > 0:
                nextX, nextY = self.startPositions.pop(0)
                self.scope.stage.MoveTo(0, nextX)
                self.scope.stage.MoveTo(1, nextY)
                self.scope.stage.moving = [1,1]
            else: #gone through all start positions -> we're done
                #self.scope.frameWrangler.WantFrameNotification.remove(self.OnTick)
                #self.scope.frameWrangler.WantFrameGroupNotification.remove(self.updateView)
                self.scope.frameWrangler.onFrame.disconnect(self.OnTick)
                self.scope.frameWrangler.onFrameGroup.disconnect(self.updateView)
                #View3D(self.data, title='tiled image')
            self.scope.frameWrangler.start()

        else: #we've got to the next starting position - fire off next run
            xp = self.scope.stage.GetPos(0)
            yp = self.scope.stage.GetPos(1)
            
            self.j = int((xp - self.rect[0])/self.pixelsize)
            self.i = int((yp - self.rect[1])/self.pixelsize)

            #print((self.i, self.j, self.data.shape))

            nextY = self.endYPositions.pop(0)

            if nextY > yp: #going forward
                self.dir = 1
            else: #going backwards
                self.dir = -1

            self.yspeed = self.ystep

            self.scope.frameWrangler.stop()
            #self.scope.frameWrangler.start()
            self.scope.frameWrangler.purge()
            self.scope.stage.MoveTo(1, nextY)
            self.runInProgress = True
            self.scope.stage.moving = [1,1]
            #self.scope.frameWrangler.stop()
            self.scope.frameWrangler.start()




class fastRectTiler(fastTiler):
    def __init__(self, scope, rect, xstep = .00005, ystep=10.5, pixelsize=PIXELSIZE):
        self.rect = rect
        self.xstep = xstep
        self.pixelsize = pixelsize

        fastTiler.__init__(self, scope, ystep=ystep, pixelsize=pixelsize)
        #print((self.startPositions))

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







