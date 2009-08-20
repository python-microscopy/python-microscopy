import scipy
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types

from PYME.Analysis import twist

import PYME.Analysis.twoColour as twoColour

#from PYME.Analysis.cModels.gauss_app import *
from PYME.PSFGen.ps_app import *


#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)


IntXVals = None
IntYVals = None
IntZVals = None

interpModel = None
interpModelName = None

dx = None
dy = None
dz = None

def genTheoreticalModel(md):
    global IntXVals, IntYVals, IntZVals, interpModel, dx, dy, dz

    if not dx == md.voxelsize.x*1e3 and not dy == md.voxelsize.y*1e3 and not dz == md.voxelsize.z*1e3:

        IntXVals = 1e3*md.voxelsize.x*scipy.mgrid[-20:20]
        IntYVals = 1e3*md.voxelsize.y*scipy.mgrid[-20:20]
        IntZVals = 1e3*md.voxelsize.z*scipy.mgrid[-20:20]

        dx = md.voxelsize.x*1e3
        dy = md.voxelsize.y*1e3
        dz = md.voxelsize.z*1e3

        P = scipy.arange(0,1.01,.01)

        interpModel = genWidefieldPSF(IntXVals, IntYVals, IntZVals, P,1e3, 0, 0, 0, 2*scipy.pi/525, 1.47, 10e3)

        interpModel = interpModel/interpModel.max() #normalise to 1

def setModel(modName, md):
    global IntXVals, IntYVals, IntZVals, interpModel, interpModelName, dx, dy, dz

    if not modName == interpModelName:
        mod = numpy.load(modName)
        interpModelName = modName

        IntXVals = 1e3*md.voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
        IntYVals = 1e3*md.voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
        IntZVals = 1e3*md.voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

        dx = md.voxelsize.x*1e3
        dy = md.voxelsize.y*1e3
        dz = md.voxelsize.z*1e3

        interpModel = mod

        interpModel = interpModel/interpModel.max() #normalise to 1

        twist.twistCal(interpModel, IntXVals, IntYVals, IntZVals)

#def interp(X, Y, Z):
#    X = scipy.array(X).reshape(-1)
#    Y = scipy.array(Y).reshape(-1)
#    Z = scipy.array(Z).reshape(-1)
#
#    ox = X[0]
#    oy = Y[0]
#    oz = Z[0]
#
#    rx = (ox % dx)/dx
#    ry = (oy % dy)/dy
#    rz = (oz % dz)/dz
#
#    fx = 20 + int(ox/dx)
#    fy = 20 + int(oy/dy)
#    fz = 20 + int(oz/dz)
#
#    #print fx
#    #print rx, ry, rz
#
#    xl = len(X)
#    yl = len(Y)
#    zl = len(Z)
#
#    #print xl
#
#    m000 = interpModel[fx:(fx+xl),fy:(fy+yl),fz:(fz+zl)]
#    m100 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),fz:(fz+zl)]
#    m010 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),fz:(fz+zl)]
#    m110 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),fz:(fz+zl)]
#
#    m001 = interpModel[fx:(fx+xl),fy:(fy+yl),(fz+1):(fz+zl+1)]
#    m101 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),(fz+1):(fz+zl+1)]
#    m011 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),(fz+1):(fz+zl+1)]
#    m111 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),(fz+1):(fz+zl+1)]
#
#    #print m000.shape
#
##    m = scipy.sum([((1-rx)*(1-ry)*(1-rz))*m000, ((rx)*(1-ry)*(1-rz))*m100, ((1-rx)*(ry)*(1-rz))*m010, ((rx)*(ry)*(1-rz))*m110,
##        ((1-rx)*(1-ry)*(rz))*m001, ((rx)*(1-ry)*(rz))*m101, ((1-rx)*(ry)*(rz))*m011, ((rx)*(ry)*(rz))*m111], 0)
#
#    m = ((1-rx)*(1-ry)*(1-rz))*m000 + ((rx)*(1-ry)*(1-rz))*m100 + ((1-rx)*(ry)*(1-rz))*m010 + ((rx)*(ry)*(1-rz))*m110+((1-rx)*(1-ry)*(rz))*m001+ ((rx)*(1-ry)*(rz))*m101+ ((1-rx)*(ry)*(rz))*m011+ ((rx)*(ry)*(rz))*m111
#    #print m.shape
#    return m

def interp(X, Y, Z):
    X = atleast_1d(X)
    Y = atleast_1d(Y)
    Z = atleast_1d(Z)

    ox = X[0]
    oy = Y[0]
    oz = Z[0]

    rx = (ox % dx)/dx
    ry = (oy % dy)/dy
    rz = (oz % dz)/dz

    fx = int(len(IntXVals)/2) + int(ox/dx)
    fy = int(len(IntYVals)/2) + int(oy/dy)
    fz = int(len(IntZVals)/2) + int(oz/dz)

    #print fx
    #print rx, ry, rz

    xl = len(X)
    yl = len(Y)
    zl = len(Z)

    #print xl

    m000 = interpModel[fx:(fx+xl),fy:(fy+yl),fz:(fz+zl)]
    m100 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),fz:(fz+zl)]
    m010 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),fz:(fz+zl)]
    m110 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),fz:(fz+zl)]

    m001 = interpModel[fx:(fx+xl),fy:(fy+yl),(fz+1):(fz+zl+1)]
    m101 = interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),(fz+1):(fz+zl+1)]
    m011 = interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),(fz+1):(fz+zl+1)]
    m111 = interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),(fz+1):(fz+zl+1)]

    #print m000.shape

#    m = scipy.sum([((1-rx)*(1-ry)*(1-rz))*m000, ((rx)*(1-ry)*(1-rz))*m100, ((1-rx)*(ry)*(1-rz))*m010, ((rx)*(ry)*(1-rz))*m110,
#        ((1-rx)*(1-ry)*(rz))*m001, ((rx)*(1-ry)*(rz))*m101, ((1-rx)*(ry)*(rz))*m011, ((rx)*(ry)*(rz))*m111], 0)

    m = ((1-rx)*(1-ry)*(1-rz))*m000 + ((rx)*(1-ry)*(1-rz))*m100 + ((1-rx)*(ry)*(1-rz))*m010 + ((rx)*(ry)*(1-rz))*m110+((1-rx)*(1-ry)*(rz))*m001+ ((rx)*(1-ry)*(rz))*m101+ ((1-rx)*(ry)*(rz))*m011+ ((rx)*(ry)*(rz))*m111
    #print m.shape
    return m

def f_Interp3d(p, X, Y, Z, P, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b

    dx = (interpModel.shape[0] - len(X))/2 - 2
    xm = len(X)/2

    dy = (interpModel.shape[1] - len(Y))/2 - 2
    ym = len(Y)/2

    #print X[0] - x0, Y[0] - y0, Z[0] - z0 , 'o', IntZVals[3]

    x0 = min(max(x0, X[xm - dx]), X[dx + xm])
    y0 = min(max(y0, Y[ym - dy]), Y[dy + ym])
    z0 = min(max(z0, Z[0] + IntZVals[2]), Z[0] + IntZVals[-2])

    #print X[0] - x0, Y[0] - y0, Z[0] - z0

    #print Z[0] - z0

    g1 = interp(X - x0 + 1, Y - y0 + 1, Z[0] - z0 + 1)*A + b

    #print g1.shape

    return g1


def f_PSF3d(p, X, Y, Z, P, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b
    g1 = genWidefieldPSF(X, Y, Z[0], P,A*1e3, x0, y0, z0, *args) + b

    return g1

class PSFFitResult:
    def __init__(self, fitResults, metadata, slicesUsed=None, resultCode=None, fitErr=None):
        self.fitResults = fitResults
        self.metadata = metadata
        self.slicesUsed = slicesUsed
        self.resultCode=resultCode
        self.fitErr = fitErr

    def A(self):
        return self.fitResults[0]

    def x0(self):
        return self.fitResults[1]

    def y0(self):
        return self.fitResults[2]

    def z0(self):
        return self.fitResults[3]

    def background(self):
        return self.fitResults[4]

    def renderFit(self):
        #X,Y = scipy.mgrid[self.slicesUsed[0], self.slicesUsed[1]]
        #return f_gauss2d(self.fitResults, X, Y)
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[self.slicesUsed[0]]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[self.slicesUsed[1]]
        Z = 1e3*self.metadata.voxelsize.z*scipy.mgrid[self.slicesUsed[2]]
        P = scipy.arange(0,1.01,.1)
        return f_PSF3d(self.fitResults, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)
        #pass

def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def PSFFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else:
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed)], dtype=fresultdtype)

		

class PSFFitFactory:
    def __init__(self, data, metadata, fitfcn=f_Interp3d, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. '''
        self.data = data
        self.metadata = metadata
        self.background = background
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if type(fitfcn) == types.FunctionType: #single function provided - use numerically estimated jacobian
            self.solver = FitModelWeighted_
        else: #should be a tuple containing the fit function and its jacobian
            self.solver = FitModelWeightedJac
        if fitfcn == f_Interp3d:
            if 'PSFFile' in metadata.getEntryNames():
                setModel(metadata.PSFFile, metadata)
            else:
                genTheoreticalModel(metadata)
		
        
    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

        #average in z
        #dataMean = dataROI.mean(2) - self.metadata.CCD.ADOffset

        #generate grid to evaluate function on        
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[xslice]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]


        Z = array([0]).astype('f')
        P = scipy.arange(0,1.01,.01)

        #print DeltaX
        #print DeltaY

        #estimate some start parameters...
        A = dataROI.max() - dataROI.min() #amplitude

        #figure()
        #imshow(dataROI[:,:,1], interpolation='nearest')

        #print Ag
        #print Ar

        x0 =  X.mean()
        y0 =  Y.mean()
        z0 = 200.0

        ta = twist.calcTwist(dataROI, X-x0, Y - y0)
        z0 = -twist.getZ(ta)

        #print z0

        startParameters = [A, x0, y0, z0, dataROI.min()]


        #estimate errors in data
        nSlices = 1#dataROI.shape[2]

        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*dataROI)/self.metadata.Camera.ElectronsPerCount

#        figure(4)
#        clf()
#        imshow(dataROI.squeeze())
#        colorbar()
#
#        figure(5)
#        clf()
#        imshow(self.fitfcn(startParameters, X, Y, Z, P).squeeze())
#        colorbar()
#
#        figure(6)
#        clf()
#        imshow(sigma.squeeze())
#        colorbar()

        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)

        #print infodict
        #print cov_x
        #print mesg

#        figure(7)
#        clf()
#        imshow(infodict['fjac'].reshape([len(X), len(Y), -1]).reshape([len(X), -1] ) == 0, interpolation='nearest')
#        colorbar()
#
#        figure(8)
#        clf()
#        imshow(infodict['fvec'].reshape([len(X), len(Y)]), interpolation='nearest')
#        colorbar()

        fitErrors=None
        try:
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors)

    def FromPoint(self, x, y, z=None, roiHalfSize=15, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()

        x = round(x)
        y = round(y)

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]),
            max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 0:2]
        

#so that fit tasks know which class to use
FitFactory = PSFFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
