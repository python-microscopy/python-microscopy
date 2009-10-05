#!/usr/bin/python

##################
# sim1.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wormlike2
wc = wormlike2.wormlikeChain(2000)

import fluor
fluors = [fluor.fluorophore(wc.xp[i], wc.yp[i], wc.zp[i], fluor.createSimpleTransitionMatrix(), [1, 30]) for i in range(len(wc.yp))]
scope.cam.fluors=fluors


import remFit6
import Pyro
tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')


def postTask(hi=None):
    im = example.CDataStack_AsArray(scope.pa.ds,0)[:,:,0]
    t = remFit6.fitTask(im, 5)
    tq.postTask(t)

scope.pa.WantFrameNotification.append(postTask)
    
from pylab import *
import matplotlib.axes3d as p3
figure(2)
ax = p3.Axes3DI(gcf())
ax.plot3D([p[0] for p in dsc.points], [p[1] for p in dsc.points], [p[2] for p in dsc.points], '.')
show()


for i in range(100):
    t = tq.getCompletedTask()
    if not (t ==None):
        if len(t.results) > 0:
            res.append(t.results[:])
    else:
        print 'Got %d results, Queue is empty' % i
        break
plot([r[0][0] for r in res], [r[0][1] for r in res], '.', hold=False)
