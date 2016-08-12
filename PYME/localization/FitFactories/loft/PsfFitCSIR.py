#!/usr/bin/python

##################
# PsfFitCSIR.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy
#from scipy.signal import interpolate
import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types
import cPickle

from PYME.localization import twist

#import PYME.Analysis.points.twoColour as twoColour

#from PYME.localization.cModels.gauss_app import *
from PYME.Analysis.PSFGen.ps_app import *
from PYME.ParallelTasks.relativeFiles import getFullExistingFilename


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

         #do the spline filtering here rather than in interpolation
        interpModel = ndimage.spline_filter(interpModel)

def setModel(modName, md):
    global IntXVals, IntYVals, IntZVals, interpModel, interpModelName, dx, dy, dz

    if not modName == interpModelName:
        mf = open(getFullExistingFilename(modName), 'rb')
        mod, voxelsize = cPickle.load(mf)
        mf.close()
        
        interpModelName = modName

        #if not voxelsize.x == md.voxelsize.x:
        #    raise RuntimeError("PSF and Image voxel sizes don't match")

        IntXVals = 1e3*voxelsize.x*mgrid[-(mod.shape[0]/2.):(mod.shape[0]/2.)]
        IntYVals = 1e3*voxelsize.y*mgrid[-(mod.shape[1]/2.):(mod.shape[1]/2.)]
        IntZVals = 1e3*voxelsize.z*mgrid[-(mod.shape[2]/2.):(mod.shape[2]/2.)]

        dx = voxelsize.x*1e3
        dy = voxelsize.y*1e3
        dz = voxelsize.z*1e3

        interpModel = mod

        interpModel = interpModel/interpModel.max() #normalise to 1

        #do the spline filtering here rather than in interpolation
        interpModel = ndimage.spline_filter(interpModel)

        twist.twistCal(interpModel, IntXVals, IntYVals, IntZVals)





def f_Interp3d(p, X, Y, Z, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b

#    x1 = x0/dx - floor(X.mean()/dx)
#    y1 = y0/dy - floor(Y.mean()/dy)
#    z1 = z0/dz - floor(Z.mean()/dz)

    x1 = (X - x0)/dx + len(IntXVals)/2.
    y1 = (Y - y0)/dy + len(IntYVals)/2.
    z1 = (Z- z0)/dz + len(IntZVals)/2.

    #print x1, y1, z1

    coords = array([x1, y1, z1])

    g1 = ndimage.interpolation.map_coordinates(interpModel, coords, mode='nearest', prefilter=False).squeeze()

    #print g1.shape
    #print X.shape

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
	if n is None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]), ('startParams', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')])]

def PSFFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, startParams=None):
	if slicesUsed is None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else:
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr is None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	if startParams is None:
		startParams = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed, startParams.astype('f'))], dtype=fresultdtype)

		

class PSFFitFactory:
    def __init__(self, data, metadata, fitfcn=f_Interp3d, background=None):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
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
        X,Y,Z = scipy.mgrid[xslice, yslice, :1]
        #X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[xslice]
        #Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]
        #Z = array([0]).astype('f')
        
        X = 1e3*self.metadata.voxelsize.x*X
        Y = 1e3*self.metadata.voxelsize.y*Y
        Z = 1e3*self.metadata.voxelsize.z*Z

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

        #ta = twist.calcTwist(dataROI, X-x0, Y - y0)
        #z0 = -twist.getZ(ta)

        #print x0, y0, z0

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
        return PSFFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, numpy.array(startParameters))

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
