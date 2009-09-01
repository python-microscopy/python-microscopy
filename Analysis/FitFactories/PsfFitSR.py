import scipy
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types
import cPickle

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
interpModelF = None

kx = None
ky = None
kz = None

dx = None
dy = None
dz = None

def genTheoreticalModel(md):
    global IntXVals, IntYVals, IntZVals, interpModel, interpModelF, dx, dy, dz, kx, ky, kz

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

        interpModelF = numpy.fft.fftn(interpModel)

        kx,ky,kz = numpy.mgrid[:interpModel.shape[0],:interpModel.shape[1],:interpModel.shape[2]]

        kx = numpy.fft.fftshift(kx - interpModel.shape[0]/2.)/interpModel.shape[0]
        ky = numpy.fft.fftshift(ky - interpModel.shape[1]/2.)/interpModel.shape[1]
        kz = numpy.fft.fftshift(kz - interpModel.shape[2]/2.)/interpModel.shape[2]

def setModel(modName, md):
    global IntXVals, IntYVals, IntZVals, interpModel, interpModelF, interpModelName, dx, dy, dz, kx, ky, kz

    if not modName == interpModelName:
        mf = open(modName, 'rb')
        mod, voxelsize = cPickle.load(mf)
        mf.close()

        interpModelName = modName

        if not voxelsize.x == md.voxelsize.x:
            raise RuntimeError("PSF and Image voxel sizes don't match")

        IntXVals = 1e3*voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
        IntYVals = 1e3*voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
        IntZVals = 1e3*voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

        dx = voxelsize.x*1e3
        dy = voxelsize.y*1e3
        dz = voxelsize.z*1e3

        interpModel = mod

        interpModel = interpModel/interpModel.max() #normalise to 1

        interpModelF = numpy.fft.fftn(interpModel)

        kx,ky,kz = numpy.mgrid[:interpModel.shape[0],:interpModel.shape[1],:interpModel.shape[2]]

        kx = numpy.fft.fftshift(kx - interpModel.shape[0]/2.)/interpModel.shape[0]
        ky = numpy.fft.fftshift(ky - interpModel.shape[1]/2.)/interpModel.shape[1]
        kz = numpy.fft.fftshift(kz - interpModel.shape[2]/2.)/interpModel.shape[2]


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


def f_Interp3d(p, X, Y, Z, P, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b
    
    #x0 = min(max(x0, X[0]), X[-1])
    #y0 = min(max(y0, Y[0]), Y[-1])
    #z0 = min(max(z0, -1.8e3), 1.8e3)

    #print Z[0] - z0

    x1 = x0/dx - floor(X.mean()/dx)
    y1 = y0/dy - floor(Y.mean()/dy)
    z1 = z0/dz - floor(Z.mean()/dz)

    m = numpy.fft.ifftn(interpModelF*numpy.exp(-2j*numpy.pi*(kx*x1 + ky*y1 + kz*z1))).real

    g1 = m[(ceil((m.shape[0]+1)/2.) - ceil(len(X)/2.)):(ceil((m.shape[0]+1)/2.) + len(X)/2), (ceil(m.shape[1]/2.) - ceil(len(Y)/2.) + 1):(floor(m.shape[1]/2.) + len(Y)/2 + 1), (floor(m.shape[2]/2.) - ceil(len(Z)/2.) + 1):(floor(m.shape[2]/2.) + len(Z)/2 + 1)]

    #print g1.shape

    return A*g1 + b


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
            self.solver = FitModelWeighted
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
        z0 = 10.0

        startParameters = [A, x0, y0, z0, dataROI.min()]


        #estimate errors in data
        nSlices = 1#dataROI.shape[2]

        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*dataROI)/self.metadata.Camera.ElectronsPerCount


        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)

        #print cov_x
        #print mesg

        fitErrors=None
        try:
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
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
