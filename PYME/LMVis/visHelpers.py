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

#!/usr/bin/python
from __future__ import print_function
import scipy
import numpy
import numpy as np
import numpy.ctypeslib

from PYME.Analysis.points.SoftRend import RenderTetrahedra
from math import floor

from PYME.IO.image import ImageBounds



#from edgeDB import genEdgeDB#, calcNeighbourDists

multiProc = False

try:
    import multiprocessing
    #import multiprocessing.sharedctypes
    from PYME.util.shmarray import shmarray
    multiProc = True
except:
    multiProc = False
        

def genEdgeDB(T):
    #make ourselves a quicker way of getting at edge info.
    edb = []
    #edb = numpy.zeros((len(T.x), 2), dtype='O')
    for i in range(len(T.x)):
        edb.append(([],[]))
        #edb[i] = ([],[])

    for i in range(len(T.edges)):
        e0, e1 = T.edges[i]
        edbe0 = edb[e0]
        edbe1 = edb[e1]
        edbe0[0].append(i)
        edbe0[1].append(e1)
        edbe1[0].append(i)
        edbe1[1].append(e0)


    return edb

mpT = {}

def calcNeighbourDistPart(di, x, y, edb, nStart, nEnd):
    for i in range(nStart, nEnd):

        dist = edb.getVertexEdgeLengths(i)

        di[i] = scipy.mean(dist)

    #return di


#if False:
#    def calcNeighbourDists(T):
#        edb = EdgeDB.EdgeDB(T, shm=True)
#
#        N = len(T.x)
#
#        di = shmarray.zeros(N)
#
#        taskSize = N/multiprocessing.cpu_count()
#        taskEdges = range(0,N, taskSize) + [N]
#        #print taskEdges
#
#        tasks = [(taskEdges[i], taskEdges[i+1]) for i in range(len(taskEdges)-1)]
#
#        x = shmarray.create_copy(T.x)
#        y = shmarray.create_copy(T.y)
#
#
#        processes = [multiprocessing.Process(target = calcNeighbourDistPart, args=(di, x, y, edb) + t) for t in tasks]
#
#        for p in processes:
#            p.start()
#
#        for p in processes:
#            p.join()
#
#        #print di[:100]
#
#
#        return di
#else:
def calcNeighbourDists(T):
    from PYME.Analysis.points import EdgeDB

    edb = EdgeDB.EdgeDB(T)
    return edb.getNeighbourDists()



def Gauss2D(Xv,Yv, A,x0,y0,s):
    from PYME.localization.cModels.gauss_app import genGauss
    r = genGauss(Xv,Yv,A,x0,y0,s,0,0,0)
    return r

def rendGauss(x, y, sx, imageBounds, pixelSize):
    """

    Parameters
    ----------
    x : ndarray
        x positions [nm]
    y : ndarray
        y positions [nm]
    sx : ndarray
        (gaussian) lateral width (sigma) [nm]
    imageBounds : PYME.IO.ImageBounds
        ImageBounds instance - range in each dimension should be an integer multiple of pixelSize. ImageBounds (x0, y0)
        and (x1, y1) correspond to the inside edge of the outer pixels.
    pixelSize : float
        size of pixels to be rendered [nm]

    Returns
    -------
    im : ndarray
        2D Gaussian rendering. Note that im[0, 0] is centered at 0.5 * [pixelSize, pixelSize] (FIXME)

    TODOS:
    
    - speed improvements? Parallelisation?
    - variable ROI size? We currently base our ROI size on the median localization/jitter error, with the parts of the
    Gaussians which extend past the ROI being dropped. This is usually not an issue, but could become one if we have a
    large range of localization precisions (or if we are using something else - e.g. neighbour distances - as sigma).
    
    """
    
    # choose a ROI size that is appropriate, and generate a padded image to render into
    sx = numpy.maximum(sx, pixelSize)
    fuzz = 3*scipy.median(sx)
    roiSize = int(fuzz/pixelSize)
    fuzz = pixelSize*roiSize

    # Gauss2D expects coordinates for pixel centres
    # FIXME - do we need the half pixel offset (this would be correct if localizations were referenced to top-left of
    # raw image, but they aren't - they are referenced to the pixel centres).
    X = numpy.arange(imageBounds.x0 - fuzz,imageBounds.x1 + fuzz, pixelSize) + 0.5*pixelSize
    Y = numpy.arange(imageBounds.y0 - fuzz,imageBounds.y1 + fuzz, pixelSize) + 0.5*pixelSize
    
    im = scipy.zeros((len(X), len(Y)), 'f')
    
    for i in range(len(x)):
        # FIXME - argmin() involves a search (expensive) - we should be able to get the nearest pixel as something like
        # ix = int(np.round((x[i] - X[0])/pixelSize))
        ix = scipy.absolute(X - x[i]).argmin()
        iy = scipy.absolute(Y - y[i]).argmin()

        sxi = sx[i]
        
        imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1/sxi, x[i],y[i],sxi)
        im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    # clip to final image size
    im = im[roiSize:-roiSize, roiSize:-roiSize]

    return im


def rend_density_estimate(x, y, imageBounds, pixelSize, N=10):
    """

    Parameters
    ----------
    x : ndarray
        x positions [nm]
    y : ndarray
        y positions [nm]
    sx : ndarray
        (gaussian) lateral width (sigma) [nm]
    imageBounds : PYME.IO.ImageBounds
        ImageBounds instance - range in each dimension should be an integer multiple of pixelSize. ImageBounds (x0, y0)
        and (x1, y1) correspond to the inside edge of the outer pixels.
    pixelSize : float
        size of pixels to be rendered [nm]

    Returns
    -------
    im : ndarray
        2D Gaussian rendering. Note that im[0, 0] is centered at 0.5 * [pixelSize, pixelSize] (FIXME)

    TODOS:

    - speed improvements? Parallelisation?
    - variable ROI size? We currently base our ROI size on the median localization/jitter error, with the parts of the
    Gaussians which extend past the ROI being dropped. This is usually not an issue, but could become one if we have a
    large range of localization precisions (or if we are using something else - e.g. neighbour distances - as sigma).

    """
    from scipy.spatial import cKDTree
    
    X = numpy.arange(imageBounds.x0, imageBounds.x1, pixelSize) + 0.5 * pixelSize
    Y = numpy.arange(imageBounds.y0, imageBounds.y1, pixelSize) + 0.5 * pixelSize
    
    im = scipy.zeros((len(X), len(Y)), 'f')
    
    
    pts = np.vstack([x, y]).T
    
    kdt = cKDTree(pts)
    n = np.arange(N)
    
    for i, xi in enumerate(X):
        for j, yi in enumerate(Y):
            d, _ = kdt.query(np.hstack((xi, yi)), N)
            im[i, j] = float(np.linalg.lstsq(np.atleast_2d(d ** 2).T, n, rcond=None)[0])
            
    
    return im
    
def rendGaussProd(x,y, sx, imageBounds, pixelSize):
    """ EXPERIMENTAL code to try and generate a log-likelihood rendering
    
    WARNING - needs lots of revision
    """
    
    sx = numpy.maximum(sx, pixelSize)
    fuzz = 6*scipy.median(sx)
    roiSize = int(fuzz/pixelSize)
    fuzz = pixelSize*(roiSize)

    #print imageBounds.x0
    #print imageBounds.x1
    #print fuzz
    #print roiSize

    #print pixelSize

    X = numpy.arange(imageBounds.x0 - fuzz,imageBounds.x1 + fuzz, pixelSize)
    Y = numpy.arange(imageBounds.y0 - fuzz,imageBounds.y1 + fuzz, pixelSize)

    #print X
    
    ctval = 1e-4
    
    l3 = numpy.log(ctval)
    #l3 = -10
    
    im = len(x)*l3*scipy.ones((len(X), len(Y)), 'd')
    print((im.min()))
    
    fac = 1./numpy.sqrt(2*numpy.pi)

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = scipy.absolute(X[1] - X[0]) 
    
    for i in range(len(x)):
        ix = scipy.absolute(X - x[i]).argmin()
        iy = scipy.absolute(Y - y[i]).argmin()
        
        if (ix > (roiSize + 1)) and (ix < (im.shape[0] - roiSize - 2)) and (iy > (roiSize+1)) and (iy < (im.shape[1] - roiSize-2)):       
            #print i, X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)]
            #imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, x[i],y[i],max(sx[i], delX))
            #print 'r'
            #imp[numpy.isnan(imp)] = ctval
            #imp = numpy.maximum(imp, ctval)
            #if imp.max() > 1:        
            #    print imp.max()
            #if not imp.min() > 1e-20:
                
            #    print imp.min()
            #imp_ = numpy.log(1.0*imp)
            
            #sxi = max(sx[i], delX)
            sxi = sx[i]
            Xi, Yi = X[(ix - roiSize):(ix + roiSize + 1)][:,None], Y[(iy - roiSize):(iy + roiSize + 1)][None,:]
            imp = numpy.log(fac/sxi) - ((Xi - x[i])**2 + (Yi -y[i])**2)/(2*sxi**2)
            print((imp.max(), imp.min(), l3, imp.shape))
            imp_ = numpy.maximum(imp, l3)
            im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp_ - l3

    im = im[roiSize:-roiSize, roiSize:-roiSize]

    return im
    



def rendTri(T, imageBounds, pixelSize, c=None, im=None, geometric_mean=False):
    from PYME.Analysis.points.SoftRend import drawTriang, drawTriangles
    xs = T.x[T.triangles]  # x posititions of vertices [nm], dimensions (# triangles, 3)
    ys = T.y[T.triangles]  # y posititions of vertices [nm], dimensions (# triangles, 3)

    if c is None:
        # We didn't pass anything in for c - use 1/ area of triangle
        a01 = numpy.vstack((xs[:, 0] - xs[:, 1], ys[:, 0] - ys[:, 1])).T
        a02 = numpy.vstack((xs[:, 0] - xs[:, 2], ys[:, 0] - ys[:, 2])).T
        a12 = numpy.vstack((xs[:, 1] - xs[:, 2], ys[:, 1] - ys[:, 2])).T

        a_ = ((a01 * a01).sum(1))
        b_ = ((a02 * a02).sum(1))
        b2_ = ((a12 * a12).sum(1))
        # use the median edge length^2 as a proxy for area (this avoids "slithers" getting really bright)
        c = 0.5*numpy.median([b_, a_, b2_], 0)
 
        #c_neighbours = c[T.triangle_neighbors].sum(1)
        #c = 1.0/(c + c_neighbours + 1)
        
        #c = numpy.maximum(c, pixelsize**2) #try to avoid spikes


    # Changed pre-factor 4/3/2019 so that image is calibrated in localizations/um^2 rather than
    # localizations per nm^2 in the hope that this will play better with other software (e.g. ImageJ)
    if geometric_mean:
        # calculate the mean areas first, then invert
        c = c
    else:
        # default - arithmetic mean (of density)
        c = 1e6/(c + 1)
    
    if im is None:
        print('Some thing is wrong - we should already have allocated memory')
        sizeX = (imageBounds.x1 - imageBounds.x0) / pixelSize
        sizeY = (imageBounds.y1 - imageBounds.y0) / pixelSize
        
        im = numpy.zeros((sizeX, sizeY))

    # convert vertices [nm] to pixel position in output image (still floating point)
    xs = (xs - imageBounds.x0) / pixelSize
    ys = (ys - imageBounds.y0) / pixelSize

    # NOTE 1: drawTriangles truncates co-ordinates to the nearest pixel on the left.
    # NOTE 2: this truncation means that nothing is drawn for triangles < 1 pixel
    # NOTE 3: triangles which would intersect with the edge of the image are discarded
    drawTriangles(im, xs, ys, c)

    return im
    


def rendJitTri(im, x, y, jsig, mcp, imageBounds, pixelSize, n=1, seed=None):
    from matplotlib import tri
    np.random.seed(seed)
    
    for i in range(int(n)):
        Imc = scipy.rand(len(x)) < mcp
        
        if isinstance(jsig, numpy.ndarray):
            #print((jsig.shape, Imc.shape))
            jsig2 = jsig[Imc]
        else:
            jsig2 = float(jsig)
        T = tri.Triangulation(x[Imc] +  jsig2*scipy.randn(Imc.sum()), y[Imc] +  jsig2*scipy.randn(Imc.sum()))

        rendTri(T, imageBounds, pixelSize, im=im, geometric_mean=False)
        
    im [:20, 0] += scipy.rand(20) #Create signature for ImageID - TODO - fix fileID code so that this is not necessary
        
    #reseed the random number generator so that anything subsequent does not become deterministic (if we specified seeds)
    np.random.seed(None)


def _rend_jit_tri_geometric(im, x, y, jsig, mcp, imageBounds, pixelSize, n=1, seed=None):
    from matplotlib import tri
    np.random.seed(seed)

    #im_ = numpy.zeros(im.shape, 'f')
    
    for i in range(int(n)):
        #im_ *= 0
        #im_ = numpy.zeros(im.shape, 'f')
        Imc = scipy.rand(len(x)) < mcp
        
        if isinstance(jsig, numpy.ndarray):
            #print((jsig.shape, Imc.shape))
            jsig2 = jsig[Imc]
        else:
            jsig2 = float(jsig)
        T = tri.Triangulation(x[Imc] + jsig2 * scipy.randn(Imc.sum()), y[Imc] + jsig2 * scipy.randn(Imc.sum()))
        
        rendTri(T, imageBounds, pixelSize, im=im, geometric_mean=True)
        
        #im[:] = (im + (im_))# + 1e9*(im_ <=0)))[:]
    
    im[:20, 0] += scipy.rand(20) #Create signature/watermark for ImageID - TODO - fix fileID code so that this is no longer necessary
    
    #reseed the random number generator so that anything subsequent does not become deterministic (if we specified seeds)
    np.random.seed(None)
    
def _generate_subprocess_seeds(preferred_n_tasks = 1, mdh=None, seeds=None):
    """
    Generate seeds for each rendering task, or pass through a given array of seeds for deterministically recreating a
    previous rendering.
    
    Parameters
    ----------
    preferred_n_tasks : Number of tasks to generate seeds for (if seeds==None). Generally the number of processor cores.
    mdh : [optional] metadata handler to store seeds to
    seeds : [optional] supplied seeds if we want to strictly reconstruct a previously generated image

    Returns
    -------
    
    seeds : an array of seeds

    """
    if seeds is None:
        seeds = np.random.randint(0, np.iinfo(np.int32).max, preferred_n_tasks)
    
    if not mdh is None:
        mdh['Rendering.RandomSeeds'] = [int(s) for s in seeds]
   
    return seeds

def _iterations_per_task(n, nTasks):
    """
    Divide the iterations to be performed (deterministically) across tasks.
     
    Parameters
    ----------
    n : int, the number of iterations
    nTasks : the number of tasks

    Returns
    -------
    
    an array with the number of iterations performed for each task

    """
    

    # generate tasks for each seed. The tasks array contains the number of iterations that that CPU should perform
    tasks = (n / nTasks) * numpy.ones(nTasks, 'i')
    tasks[:(n % nTasks)] += 1
    
    return tasks
    

def rendJitTriang(x,y,n,jsig, mcp, imageBounds, pixelSize, seeds=None, geometric_mean=True, mdh=None):
    """

    Parameters
    ----------
    x : ndarray
        x positions [nm]
    y : ndarray
        y positions [nm]
    n : number of jittered renderings to average into final rendering
    jsig : ndarray (or scalar float)
        standard deviations [nm] of normal distributions to sample when jittering for each point
    mcp : float
        Monte Carlo sampling probability (0, 1]
    imageBounds : PYME.IO.ImageBounds
        ImageBounds instance - range in each dimension should ideally be an integer multiple of pixelSize.
    pixelSize : float
        size of pixels to be rendered [nm]
    seeds : ndarray
        [optional] supplied seeds if we want to strictly reconstruct a previously generated image
    geometric_mean : bool
        [optional] Flag to scale intensity by geometric mean (True) or [localizations / um^2] (False)
    mdh: PYME.IO.MetaDataHandler.MDHandlerBase or subclass
        [optional] metadata handler to store seeds to

    Returns
    -------
    im : ndarray
        2D Jittered Triangulation rendering.

    Notes
    -----
    Triangles which reach outside of the image bounds are dropped and not included in the rendering.
    """
    sizeX = int((imageBounds.x1 - imageBounds.x0) / pixelSize)
    sizeY = int((imageBounds.y1 - imageBounds.y0) / pixelSize)
    
    if geometric_mean:
        fcn = _rend_jit_tri_geometric
    else:
        fcn = rendJitTri
    
    if multiProc and not multiprocessing.current_process().daemon:
        im = shmarray.zeros((sizeX, sizeY))

        x = shmarray.create_copy(x)
        y = shmarray.create_copy(y)
        if type(jsig) == numpy.ndarray:
            jsig = shmarray.create_copy(jsig)

        # We will generate 1 process for each seed, defaulting to generating a seed for each CPU core if seeds are not
        # passed explicitly. Rendering with explicitly passed seeds will be deterministic, but performance will not be
        # optimal unless n_seeds = n_CPUs
        seeds = _generate_subprocess_seeds(multiprocessing.cpu_count(), mdh, seeds)
        iterations = _iterations_per_task(n, len(seeds))

        processes = [multiprocessing.Process(target = fcn, args=(im, x, y, jsig, mcp, imageBounds, pixelSize, nIt, s)) for nIt, s in zip(iterations, seeds)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        im = numpy.zeros((sizeX, sizeY))

        # Technically we could just call fcn( ....,n), but we replicate the logic above and divide into groups of tasks
        # so that we can reproduce a previously generated image
        seeds = _generate_subprocess_seeds(1, mdh, seeds)
        iterations = _iterations_per_task(n, len(seeds))
        
        for nIt, s in zip(iterations, seeds):
            # NB - in normal usage, this loop only evaluates once, with nIt=n
            fcn(im, x, y, jsig, mcp, imageBounds, pixelSize, nIt, seed=s)
    
    if geometric_mean:
        return (1.e6/(im/n + 1))*(im > n)
    else:
        return im/n

###########
#with weighted averaging

def rendTri2(T, imageBounds, pixelSize, c=None, im=None, im1=None):
    from PYME.Analysis.points.SoftRend import drawTriang, drawTriangles
    xs = T.x[T.triangles]
    ys = T.y[T.triangles]
    
    a = numpy.vstack((xs[:, 0] - xs[:, 1], ys[:, 0] - ys[:, 1])).T
    b = numpy.vstack((xs[:, 0] - xs[:, 2], ys[:, 0] - ys[:, 2])).T
    b2 = numpy.vstack((xs[:, 1] - xs[:, 2], ys[:, 1] - ys[:, 2])).T
    
    #area of triangle
    #c = 0.5*numpy.sqrt((b*b).sum(1) - ((a*b).sum(1)**2)/(a*a).sum(1))*numpy.sqrt((a*a).sum(1))
    
    #c = 0.5*numpy.sqrt((b*b).sum(1)*(a*a).sum(1) - ((a*b).sum(1)**2))
    
    #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))
    
    c = numpy.abs(a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0])
    
    if c is None:
        if numpy.version.version > '1.2':
            c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)], 0)
        else:
            c = numpy.median([(b * b).sum(1), (a * a).sum(1), (b2 * b2).sum(1)])
            
            #c = c*c/1e6
    
    a_ = ((a * a).sum(1))
    b_ = ((b * b).sum(1))
    b2_ = ((b2 * b2).sum(1))
    #c_neighbours = c[T.triangle_neighbors].sum(1)
    #c = 1.0/(c + c_neighbours + 1)
    #c = numpy.maximum(c, self.pixelsize**2)
    #c = 1.0/(c + 1)
    
    sizeX = int((imageBounds.x1 - imageBounds.x0) / pixelSize)
    sizeY = int((imageBounds.y1 - imageBounds.y0) / pixelSize)
    
    xs = (xs - imageBounds.x0) / pixelSize
    ys = (ys - imageBounds.y0) / pixelSize
    
    if im is None:
        im = numpy.zeros((sizeX, sizeY))
        im1 = numpy.zeros_like(im)
    
    drawTriangles(im, xs, ys, c * c * c)
    drawTriangles(im1, xs, ys, c * c * c * c)
    
    return im, im1

def rendJitTri2(im, im1, x, y, jsig, mcp, imageBounds, pixelSize, n=1):
    from matplotlib import tri
    scipy.random.seed()
    
    for i in range(n):
        Imc = scipy.rand(len(x)) < mcp
        
        if isinstance(jsig, numpy.ndarray):
            #print((jsig.shape, Imc.shape))
            jsig2 = jsig[Imc]
        else:
            jsig2 = float(jsig)
            
        T = tri.Triangulation(x[Imc] +  jsig2*scipy.randn(Imc.sum()), y[Imc] +  jsig2*scipy.randn(Imc.sum()))

        rendTri2(T, imageBounds, pixelSize, im=im, im1=im1)



def rendJitTriang2(x,y,n,jsig, mcp, imageBounds, pixelSize):
    sizeX = int((imageBounds.x1 - imageBounds.x0) / pixelSize)
    sizeY = int((imageBounds.y1 - imageBounds.y0) / pixelSize)
    
    if multiProc and not multiprocessing.current_process().daemon:
        im = shmarray.zeros((sizeX, sizeY))
        im1 = shmarray.zeros((sizeX, sizeY))

        x = shmarray.create_copy(x)
        y = shmarray.create_copy(y)
        if type(jsig) == numpy.ndarray:
            jsig = shmarray.create_copy(jsig)


        nCPUs = multiprocessing.cpu_count()

        tasks = int(n / nCPUs) * numpy.ones(nCPUs, 'i')
        tasks[:int(n%nCPUs)] += 1

        processes = [multiprocessing.Process(target = rendJitTri2, args=(im, im1, x, y, jsig, mcp, imageBounds, pixelSize, nIt)) for nIt in tasks]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        im = numpy.zeros((sizeX, sizeY))
        im1 = numpy.zeros((sizeX, sizeY))

        rendJitTri2(im, im1, x, y, jsig, mcp, imageBounds, pixelSize, n)

    imn =  im/(im1+1) #n
    return imn


def rendJTet(im, x,y,z,jsig, jsigz, mcp, n):
    for i in range(n):
        scipy.random.seed()

        Imc = scipy.rand(len(x)) < mcp
        if isinstance(jsig, numpy.ndarray):
            #print((jsig.shape, Imc.shape))
            jsig_ = jsig[Imc]
            jsigz_ = jsigz[Imc]
        else:
            jsig_= jsig
            jsigz_ = jsigz

        #gen3DTriangs.renderTetrahedra(im, x[Imc]+ jsig*scipy.randn(Imc.sum()), y[Imc]+ jsig*scipy.randn(Imc.sum()), z[Imc]+ jsigz*scipy.randn(Imc.sum()), scale = [1,1,1], pixelsize=[1,1,1])
        p = numpy.hstack(((x[Imc]+ jsig_*scipy.randn(Imc.sum()))[:, None], (y[Imc]+ jsig_*scipy.randn(Imc.sum()))[:, None], (z[Imc]+ jsigz_*scipy.randn(Imc.sum()))[:, None]))
        #print((p.shape))
        RenderTetrahedra(p, im)

#if multiProc:

def rendJitTet(x,y,z,n,jsig, jsigz, mcp, imageBounds, pixelSize, sliceSize=100):
    # FIXME - signature now differs from visHelpersMin
    
    #import gen3DTriangs
    sizeX = int((imageBounds.x1 - imageBounds.x0) / pixelSize)
    sizeY = int((imageBounds.y1 - imageBounds.y0) / pixelSize)
    sizeZ = int((imageBounds.z1 - imageBounds.z0) / sliceSize)

    # convert from [nm] to [pixels]
    x = (x - imageBounds.x0) / pixelSize
    y = (y - imageBounds.y0) / pixelSize
    z = (z - imageBounds.z0) / sliceSize

    jsig = jsig / pixelSize
    jsigz = jsigz / sliceSize
    
    
    if multiProc and not multiprocessing.current_process().daemon:
        im = shmarray.zeros((sizeX, sizeY, sizeZ), order='F')

        x = shmarray.create_copy(x)
        y = shmarray.create_copy(y)
        z = shmarray.create_copy(z)

        if type(jsig) == numpy.ndarray:
            jsig = shmarray.create_copy(jsig)

        if type(jsigz) == numpy.ndarray:
            jsigz = shmarray.create_copy(jsigz)


        nCPUs = multiprocessing.cpu_count()

        tasks = int(n / nCPUs) * numpy.ones(nCPUs, 'i')
        tasks[:int(n % nCPUs)] += 1

        processes = [multiprocessing.Process(target = rendJTet, args=(im, y, x,z, jsig, jsigz, mcp, nIt)) for nIt in tasks]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return im/n

    else:
        im = numpy.zeros((sizeX, sizeY, sizeZ), order='F')

        rendJTet(im, y, x, z, jsig, jsigz, mcp, n)

        return im/n


def rendHist(x,y, imageBounds, pixelSize):
    X = numpy.arange(imageBounds.x0, imageBounds.x1 + 1.01*pixelSize, pixelSize)
    Y = numpy.arange(imageBounds.y0, imageBounds.y1 + 1.01*pixelSize, pixelSize)
    
    im, edx, edy = scipy.histogram2d(x,y, bins=(X,Y))

    return im

def rendHist3D(x,y,z, imageBounds, pixelSize,sliceSize=100):
    X = numpy.arange(imageBounds.x0, imageBounds.x1 + 1.01*pixelSize, pixelSize)
    Y = numpy.arange(imageBounds.y0, imageBounds.y1 + 1.01*pixelSize, pixelSize)
    Z = numpy.arange(imageBounds.z0,imageBounds.z1 + 1.01*sliceSize, sliceSize)

    im, ed = scipy.histogramdd([x,y, z], bins=(X,Y,Z))

    return im


def Gauss3d(X, Y, Z, x0, y0, z0, wxy, wz):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    #A, x0, y0, z0, wxy, wz, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b

    #print X.shape

    return scipy.exp(-((X[:,None]-x0)**2 + (Y[None,:] - y0)**2)/(2*wxy**2) - ((Z-z0)**2)/(2*wz**2))/((2*scipy.pi*wxy**2)*scipy.sqrt(2*scipy.pi*wz**2))

def rendGauss3D(x,y, z, sx, sz, imageBounds, pixelSize, zb, sliceSize=100):
    from PYME.localization.cModels.gauss_app import genGauss3D
    sx = numpy.maximum(sx, pixelSize)
    fuzz = 3*scipy.median(sx)
    roiSize = int(fuzz/pixelSize)
    fuzz = pixelSize*roiSize

    #print imageBounds.x0
    #print imageBounds.x1
    #print fuzz

    #print pixelSize

    X = numpy.arange(imageBounds.x0 - fuzz,imageBounds.x1 + fuzz, pixelSize)
    Y = numpy.arange(imageBounds.y0 - fuzz,imageBounds.y1 + fuzz, pixelSize)
    Z = numpy.arange(zb[0], zb[1], sliceSize)

    #print X

    im = scipy.zeros((len(X), len(Y), len(Z)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = scipy.absolute(X[1] - X[0])
    delZ = scipy.absolute(Z[1] - Z[0])

    #for zn in range(len(Z)):
    for i in range(len(x)):
        ix = scipy.absolute(X - x[i]).argmin()
        iy = scipy.absolute(Y - y[i]).argmin()
        iz = scipy.absolute(Z - z[i]).argmin()
        
        dz = int(round(2*sz[i]/delZ))

        iz_min = max(iz - dz, 0)
        iz_max = min(iz + dz + 1, len(Z))


        imp = genGauss3D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],Z[iz_min:iz_max], 1.0e3,x[i],y[i],z[i], sx[i],max(sz[i], sliceSize))
        #print imp.shape
        #print im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1), zn].shape
        im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1), iz_min:iz_max] += imp

    im = im[roiSize:-roiSize, roiSize:-roiSize, :]

    return im


def rendVoronoi(x, y, imageBounds, pixelSize):
    from matplotlib import tri
    from PYME.Analysis.points.SoftRend import drawTriang, drawTriangles
    from PYME.recipes.pointcloud import Tesselation
    sizeX = int((imageBounds.x1 - imageBounds.x0) / pixelSize)
    sizeY = int((imageBounds.y1 - imageBounds.y0) / pixelSize)
    
    im = np.zeros((sizeX, sizeY))
    
    #T = tri.Triangulation(x, y)
    Ts = Tesselation({'x': x, 'y': y, 'z': 0 * x}, three_d=False)
    cc = Ts.circumcentres()
    T = Ts.T

    tdb = []
    for i in range(len(x)):
        tdb.append([])

    for i in range(len(T.simplices)):
        nds = T.simplices[i]
        for n in nds:
            tdb[n].append(i)

    xs_ = None
    ys_ = None
    c_ = None

    area_colouring = True
    for i in range(len(x)):
        #get triangles around point
        impingentTriangs = tdb[i] #numpy.where(T.triangle_nodes == i)[0]
        if len(impingentTriangs) >= 3:
        
            circumcenters = cc[impingentTriangs] #get their circumcenters
        
            #add current point - to deal with edge cases
            newPts = np.array(list(circumcenters) + [[x[i], y[i]]])
        
            #re-triangulate (we could try and sort the triangles somehow, but this is easier)
            T2 = tri.Triangulation(newPts[:, 0], newPts[:, 1])
        
            #now do the same as for the standard triangulation
            xs = T2.x[T2.triangles]
            ys = T2.y[T2.triangles]
        
            a = np.vstack((xs[:, 0] - xs[:, 1], ys[:, 0] - ys[:, 1])).T
            b = np.vstack((xs[:, 0] - xs[:, 2], ys[:, 0] - ys[:, 2])).T
        
            #area of triangle
            c = 0.5 * np.sqrt((b * b).sum(1) - ((a * b).sum(1) ** 2) / (a * a).sum(1)) * np.sqrt((a * a).sum(1))
        
            #c = numpy.maximum(((b*b).sum(1)),((a*a).sum(1)))
        
            #c_neighbours = c[T.triangle_neighbors].sum(1)
            #c = 1.0/(c + c_neighbours + 1)
            c = c.sum() * np.ones(c.shape)
            c = 1.0 / (c + 1)
        
        
            #print xs.shape
            #print c.shape
        
            if xs_ is None:
                xs_ = xs
                ys_ = ys
                c_ = c
            else:
                xs_ = np.vstack((xs_, xs))
                ys_ = np.vstack((ys_, ys))
                c_ = np.hstack((c_, c))

    

    # convert vertices [nm] to pixel position in output image (still floating point)
    xs = (xs_ - imageBounds.x0) / pixelSize
    ys = (ys_ - imageBounds.y0) / pixelSize

    # NOTE 1: drawTriangles truncates co-ordinates to the nearest pixel on the left.
    # NOTE 2: this truncation means that nothing is drawn for triangles < 1 pixel
    # NOTE 3: triangles which would intersect with the edge of the image are discarded
    drawTriangles(im, xs, ys, c_)
    
    return im