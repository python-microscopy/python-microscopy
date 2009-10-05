#!/usr/bin/python

##################
# ccdCalibrator.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python
from pylab import *
from PYME.Analysis._fithelpers import *

def sqrtMod(p, x):
    offset, gain = p
    return sqrt(gain*(x - offset))

class ccdCalibrator:
    def __init__(self, scope, numFrames = 10):
        self.scope = scope
        self.frameNum = 0
        self.numFrames = numFrames

        self.scope.pa.WantFrameNotification.append(self.Tick)
        self.frames = []

    def Tick(self, caller):
        self.frames.append((cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight())))
        self.frameNum += 1

        if self.frameNum >= self.numFrames:
            self.scope.pa.WantFrameNotification.remove(self.Tick)

            a = vstack([cSMI.CDataStack_AsArray(scope.pa.ds, i).ravel() for i in range(8)])

            am = a.mean(0)
            a_s = a.std(0)

            xb = arange(1000, 2000, 10)
            yv = zeros(xb.shape)

            syv = zeros(xb.shape)

            for i in range(len(xb)):
                as_i = a_s[(am >xb[i] - 5)*(am <= xb[i]+5)]
                yv[i] = as_i.mean()
                syv[i] = as_i.std()/sqrt(len(as_i))

            I2 = isnan(yv) == False
            r = FitModelWeighted(sqrtMod, [1100, 1], yv[I2], syv[I2], xb[I2])

            self.offset = r[0][0]
            self.gain = r[0][1]

            plot(yv[I2],syv[I2], 'x', yv[I2], sqrtMod(r[0], yv[I2]))
