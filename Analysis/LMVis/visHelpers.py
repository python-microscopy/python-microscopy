#!/usr/bin/python

##################
# visHelpers.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python
import scipy
import numpy
from PYME.Analysis.cModels.gauss_app import *

class ImageBounds:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @classmethod
    def estimateFromSource(cls, ds):
        return cls(ds['x'].min(),ds['y'].min(),ds['x'].max(), ds['y'].max() )

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0


class dummy:
    pass

class GeneratedImage:
    def __init__(self, img, imgBounds, pixelSize):
        self.img = img
        self.imgBounds = imgBounds

        self.pixelSize = pixelSize

    def save(self, filename):
        import Image
        #save using PIL - because we're using float pretty much only tif will work
        im = Image.fromarray(self.img.astype('f'), 'F')

        im.tag = dummy()
        #set up resolution data - unfortunately in cm as TIFF standard only supports cm and inches
        res_ = int(1e-2/(self.pixelSize*1e-9))
        im.tag.tagdata={296:(3,), 282:(res_,1), 283:(res_,1)}

        im.save(filename)

        

def genEdgeDB(T):
    #make ourselves a quicker way of getting at edge info.
    edb = []
    for i in range(len(T.x)):
        edb.append(([],[]))

    for i in range(len(T.edge_db)):
        e = T.edge_db[i]
        edb[e[0]][0].append(i)
        edb[e[0]][1].append(e[1])
        edb[e[1]][0].append(i)
        edb[e[1]][1].append(e[0])


    return edb

def calcNeighbourDists(T):
    edb = genEdgeDB(T)

    di = scipy.zeros(T.x.shape)

    for i in range(len(T.x)):
        incidentEdges = T.edge_db[edb[i][0]]
        #neighbourPoints = edb[i][1]

        #incidentEdges = T.edge_db[edb[neighbourPoints[0]][0]]
        #for j in range(1, len(neighbourPoints)):
        #    incidentEdges = scipy.vstack((incidentEdges, T.edge_db[edb[neighbourPoints[j]][0]]))
        dx = scipy.diff(T.x[incidentEdges])
        dy = scipy.diff(T.y[incidentEdges])

        dist = (dx**2 + dy**2)

        di[i] = scipy.mean(scipy.sqrt(dist))

    return di


def Gauss2D(Xv,Yv, A,x0,y0,s):
    r = genGauss(Xv,Yv,A,x0,y0,s,0,0,0)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    return r

def rendGauss(x,y, sx, imageBounds, pixelSize):
    fuzz = 3*scipy.median(sx)
    roiSize = fuzz/pixelSize

    #print imageBounds.x0
    #print imageBounds.x1
    #print fuzz

    #print pixelSize

    X = numpy.arange(imageBounds.x0 - fuzz,imageBounds.x1 + fuzz, pixelSize)
    Y = numpy.arange(imageBounds.y0 - fuzz,imageBounds.y1 + fuzz, pixelSize)

    #print X
    
    im = scipy.zeros((len(X), len(Y)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = scipy.absolute(X[1] - X[0]) 
    
    for i in range(len(x)):
        ix = scipy.absolute(X - x[i]).argmin()
        iy = scipy.absolute(Y - y[i]).argmin()

        
        imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, x[i],y[i],max(sx[i], delX))
        im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    im = im[roiSize:-roiSize, roiSize:-roiSize]

    return im


def rendHist(x,y, imageBounds, pixelSize):
    X = numpy.arange(imageBounds.x0,imageBounds.x1, pixelSize)
    Y = numpy.arange(imageBounds.y0,imageBounds.y1, pixelSize)
    
    im, edx, edy = scipy.histogram2d(x,y, bins=(X,Y))

    return im
