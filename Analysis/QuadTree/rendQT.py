import matplotlib
from pylab import *


def rendQT(ax, qt, maxDepth=-1, cmap=cm.hot, maxIsc = 1):
    ax.cla()
    ax.set_xlim(qt.x0,qt.x1)
    ax.set_ylim(qt.y0, qt.y1)

    lvs = qt.getLeaves(maxDepth)

    Is = [float(l.numRecords)*2**(2*l.depth) for l in lvs]

    maxI = max(Is)
    print maxI

    for l in lvs:
        c = cmap((float(l.numRecords)*2**(2*l.depth))/(maxI*maxIsc))
        ax.add_patch(matplotlib.patches.Rectangle((l.x0,l.y0), l.x1- l.x0, l.y1 - l.y0, fc= c, ec = c))

    show()

def rendQTe(ax, qt, maxDepth=-1, cmap=cm.hot, maxIsc = 1):
    ax.cla()
    ax.set_xlim(qt.x0,qt.x1)
    ax.set_ylim(qt.y0, qt.y1)

    lvs = qt.getLeaves(maxDepth)

    Is = [float(l.numRecords)*2**(2*l.depth) for l in lvs]

    maxI = max(Is)
    print maxI

    for l in lvs:
        c = cmap((float(l.numRecords)*2**(2*l.depth))/(maxI*maxIsc))
        ax.add_patch(matplotlib.patches.Rectangle((l.x0,l.y0), l.x1- l.x0, l.y1 - l.y0, fc=[1 1 1], ec = 'k'))

    show()

def rendQTa(ar, qt, maxDepth=100):
    (sX, sY) = ar.shape    
    
    maxDepth = min(maxDepth, floor(log2(min(sX, sY))))

    lvs = qt.getLeaves(maxDepth)

    dX = float(qt.x1 - qt.x0)/sX
    dY = float(qt.y1 - qt.y0)/sY

    #ensure aspect ratio is OK
    dX = max(dX, dY)
    dY = dX

    print dX

    for l in lvs:
        dx = l.x1 - l.x0
        #if dx < dX:
        #    print dx
        ar[round((l.x0 - qt.x0)/dX):round((l.x1 - qt.x0)/dX),
	   round((l.y0 - qt.y0)/dY):round((l.y1 - qt.y0)/dY)] = \
	   (float(l.numRecords)*2**(2*(l.depth - maxDepth)))

    #return ar

# normalized version: number of events per pixel, ar.sum should be equal to total number of events
# only approximate due to rounding at pixel boundaries
def rendQTan(ar, qt, maxDepth=100):
    (sX, sY) = ar.shape    
    
    maxDepth = min(maxDepth, floor(log2(min(sX, sY))))

    lvs = qt.getLeaves(maxDepth)

    dX = float(qt.x1 - qt.x0)/sX
    dY = float(qt.y1 - qt.y0)/sY

    #ensure aspect ratio is OK
    dX = max(dX, dY)
    dY = dX

    print dX

    ar[:,:] = 0 # initialize
    numevts = 0
    nevts2 = 0
    for l in lvs:
	ix0 = round((l.x0 - qt.x0)/dX)
	ix1 = round((l.x1 - qt.x0)/dX)
	iy0 = round((l.y0 - qt.y0)/dY)
	iy1 = round((l.y1 - qt.y0)/dY)
        ar[ix0:ix1,iy0:iy1] += float(l.numRecords)/float((ix1-ix0)*(iy1-iy0))
