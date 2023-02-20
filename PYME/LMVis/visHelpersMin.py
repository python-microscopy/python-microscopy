#!/usr/bin/python

##################
# visHelpers.py
#
# Copyright David Baddeley, 2009
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
##################

import scipy
import numpy

from math import floor

from PYME.Analysis.points.SoftRend import RenderTetrahedra
from PYME.Analysis.points import EdgeDB

multiProc = False

try:
    import multiprocessing
    from PYME.util.shmarray import shmarray
    multiProc = True
except:
    multiProc = False


class ImageBounds:
    def __init__(self, x0, y0, x1, y1, z0=0, z1=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    @classmethod
    def estimateFromSource(cls, ds):
        return cls(ds['x'].min(),ds['y'].min(),ds['x'].max(), ds['y'].max() )

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0




def calcNeighbourDists(T):
    edb = EdgeDB.EdgeDB(T)

    return edb.getNeighbourDists()



def rendTri(T, imageBounds, pixelSize, c=None, im=None):
    from PYME.Analysis.points.SoftRend import drawTriang, drawTriangles
    xs = T.x[T.triangles]
    ys = T.y[T.triangles]

    a = numpy.vstack((xs[:,0] - xs[:,1], ys[:,0] - ys[:,1])).T
    b = numpy.vstack((xs[:,0] - xs[:,2], ys[:,0] - ys[:,2])).T
    b2 = numpy.vstack((xs[:,1] - xs[:,2], ys[:,1] - ys[:,2])).T

    #area of triangle
    #c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))

    #c = 0.5*numpy.sqrt((b*b).sum(1)*(a*a).sum(1) - ((a*b).sum(1)**2))

    #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))

    if c is None:
        if numpy.version.version > '1.2':
            c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)], 0)
        else:
            c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)])

    #a_ = ((a*a).sum(1))
    #b_ = ((b*b).sum(1))
    #b2_ = ((b2*b2).sum(1))
    #c_neighbours = c[T.triangle_neighbors].sum(1)
    #c = 1.0/(c + c_neighbours + 1)
    #c = numpy.maximum(c, self.pixelsize**2)
    c = 1.0/(c + 1)

    sizeX = (imageBounds.x1 - imageBounds.x0)/pixelSize
    sizeY = (imageBounds.y1 - imageBounds.y0)/pixelSize

    xs = (xs - imageBounds.x0)/pixelSize
    ys = (ys - imageBounds.y0)/pixelSize

    if im is None:
        im = numpy.zeros((sizeX, sizeY))

    drawTriangles(im, xs, ys, c)

    return im


def rendJitTri(im, x, y, jsig, mcp, imageBounds, pixelSize, n=1):
    from matplotlib import tri
    for i in range(n):
        #global jParms
        #locals().update(jParms)
        numpy.random.seed()

        Imc = numpy.random.rand(len(x)) < mcp
        if type(jsig) == numpy.ndarray:
            #print jsig.shape, Imc.shape
            jsig = jsig[Imc]
        T = tri.Triangulation(x[Imc] +  jsig*numpy.random.randn(Imc.sum()), y[Imc] +  jsig*numpy.random.randn(Imc.sum()))

        #return T
        rendTri(T, imageBounds, pixelSize, im=im)




if multiProc:
    def rendJitTriang(x,y,n,jsig, mcp, imageBounds, pixelSize):
        sizeX = int((imageBounds.x1 - imageBounds.x0)/pixelSize)
        sizeY = int((imageBounds.y1 - imageBounds.y0)/pixelSize)

        im = shmarray.zeros((sizeX, sizeY))

        x = shmarray.create_copy(x)
        y = shmarray.create_copy(y)
        if type(jsig) == numpy.ndarray:
            jsig = shmarray.create_copy(jsig)


        nCPUs = multiprocessing.cpu_count()

        tasks = (n/nCPUs)*numpy.ones(nCPUs, 'i')
        tasks[:(n%nCPUs)] += 1

        processes = [multiprocessing.Process(target = rendJitTri, args=(im, x, y, jsig, mcp, imageBounds, pixelSize, nIt)) for nIt in tasks]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return im/n
else:
    def rendJitTriang(x,y,n,jsig, mcp, imageBounds, pixelSize):
        from matplotlib import delaunay
        sizeX = (imageBounds.x1 - imageBounds.x0)/pixelSize
        sizeY = (imageBounds.y1 - imageBounds.y0)/pixelSize

        im = numpy.zeros((sizeX, sizeY))

        for i in range(n):
            Imc = numpy.random.rand(len(x)) < mcp
            if type(jsig) == numpy.ndarray:
                #print jsig.shape, Imc.shape
                jsig = jsig[Imc]
            T = delaunay.Triangulation(x[Imc] +  jsig*numpy.random.randn(Imc.sum()), y[Imc] +  jsig*numpy.random.randn(Imc.sum()))
            rendTri(T, imageBounds, pixelSize, im=im)

        return im/n


def rendJitTet(x,y,z,n,jsig, jsigz, mcp, imageBounds, pixelSize, zb,sliceSize=100):
    #from PYME.Analysis.points import gen3DTriangs

    sizeX = (imageBounds.x1 - imageBounds.x0)/pixelSize
    sizeY = (imageBounds.y1 - imageBounds.y0)/pixelSize

    x = (x - imageBounds.x0)/pixelSize
    y = (y - imageBounds.y0)/pixelSize

    jsig = jsig/pixelSize
    jsigz = jsigz/sliceSize

    z = (z - zb[0])/sliceSize

    sizeZ  = floor((zb[1] + sliceSize - zb[0])/sliceSize)

    im = numpy.zeros((sizeX, sizeY, sizeZ), order='F')

    for i in range(n):
        Imc = numpy.random.rand(len(x)) < mcp
        if type(jsig) == numpy.ndarray:
            print((jsig.shape, Imc.shape))
            jsig = jsig[Imc]
            jsigz = jsigz[Imc]

        #gen3DTriangs.renderTetrahedra(im, x[Imc]+ jsig*scipy.randn(Imc.sum()), y[Imc]+ jsig*scipy.randn(Imc.sum()), z[Imc]+ jsigz*scipy.randn(Imc.sum()), scale = [1,1,1], pixelsize=[1,1,1])
        p = numpy.hstack(((x[Imc]+ jsig*numpy.random.randn(Imc.sum()))[:, None], (y[Imc]+ jsig*numpy.random.randn(Imc.sum()))[:, None], (z[Imc]+ jsigz*numpy.random.randn(Imc.sum()))[:, None]))
        #print p.shape
        RenderTetrahedra(p, im)

    return im/n


def rendHist(x,y, imageBounds, pixelSize):
    X = numpy.arange(imageBounds.x0,imageBounds.x1, pixelSize)
    Y = numpy.arange(imageBounds.y0,imageBounds.y1, pixelSize)
    
    im, edx, edy = scipy.histogram2d(x,y, bins=(X,Y))

    return im

def rendHist3D(x,y,z, imageBounds, pixelSize, zb,sliceSize=100):
    X = numpy.arange(imageBounds.x0,imageBounds.x1, pixelSize)
    Y = numpy.arange(imageBounds.y0,imageBounds.y1, pixelSize)
    Z = numpy.arange(zb[0], zb[1] + sliceSize, sliceSize)

    im, ed = scipy.histogramdd([x,y, z], bins=(X,Y,Z))

    return im


