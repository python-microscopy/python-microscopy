#!/usr/bin/python

##################
# PsfFitIR.py
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
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types
import cPickle

from PYME.Analysis._fithelpers import *
#from PYME.Analysis.FitFactories.zEstimators import astigEstimator

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

def f_Interp3d(p, interpolator, X, Y, Z, safeRegion, splitaxis, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b, r = p

    #make sure our model is big enough to stretch to our current position
#    xm = len(X)/2
#    dx = min((interpolator.shape[0] - len(X))/2, xm) - 2
#
#    ym = len(Y)/2
#    dy = min((interpolator.shape[1] - len(Y))/2, ym) - 2
#
#
#    x0 = min(max(x0, X[xm - dx]), X[dx + xm])
#    y0 = min(max(y0, Y[ym - dy]), Y[dy + ym])
#    z0 = min(max(z0, Z[0] + interpolator.IntZVals[2]), Z[0] + interpolator.IntZVals[-2])

    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0]), safeRegion[2][1])
    
    im =  interpolator.interp(X - x0 + 1, Y - y0 + 1, Z - z0 + 1)*A 
    
    #print im.shape

    if splitaxis == 'x':
        fac = 2*(r*(X < x0) + (1-r)*(X >= x0))
        im = im*fac[:,None, None]
        #print 'x', fac.shape
    else: #splitaxis == 'y'
        fac = 2*(r*(Y < y0) + (1-r)*(Y >= y0))
        #print fac.shape
        im = im*fac[None, :, None]
        
    #print im.shape
    
    return im + b


def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n



fresultdtype=[('tIndex', '<i4'),
    ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4'), ('ratio', '<f4')]),
    ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4'), ('ratio', '<f4')]) ,
    #('coiR', [('sxl', '<f4'),('sxr', '<f4'),('syu', '<f4'),('syd', '<f4')]),
    ('resultCode', '<i4'),
    ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
    ('startParams', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4'), ('ratio', '<f4')]), ('nchi2', '<f4')]

def PSFFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, startParams=None, nchi2=-1):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else:
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	if startParams == None:
		startParams = -5e3*numpy.ones(fitResults.shape, 'f')

	tIndex = metadata.tIndex

	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed, startParams.astype('f'), nchi2)], dtype=fresultdtype)


#def genFitImage(fitResults, metadata, fitfcn=f_Interp3d):
#    if fitfcn == f_Interp3d:
#        if 'PSFFile' in metadata.getEntryNames():
#            setModel(metadata.getEntry('PSFFile'), metadata)
#        else:
#            genTheoreticalModel(metadata)
#
#    xslice = slice(*fitResults['slicesUsed']['x'])
#    yslice = slice(*fitResults['slicesUsed']['y'])
#
#    X = 1e3*metadata.getEntry('voxelsize.x')*scipy.mgrid[xslice]
#    Y = 1e3*metadata.getEntry('voxelsize.y')*scipy.mgrid[yslice]
#    Z = array([0]).astype('f')
#    P = scipy.arange(0,1.01,.01)
#
#    im = fitfcn(fitResults['fitResults'], X, Y, Z, P).reshape(len(X), len(Y))
#
#    return im

def getDataErrors(im, metadata):
    dataROI = im - metadata.getEntry('Camera.ADOffset')

    return scipy.sqrt(metadata.getEntry('Camera.ReadNoise')**2 + (metadata.getEntry('Camera.NoiseFactor')**2)*metadata.getEntry('Camera.ElectronsPerCount')*metadata.getEntry('Camera.TrueEMGain')*dataROI)/metadata.getEntry('Camera.ElectronsPerCount')

		

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
        

        interpModule = metadata.Analysis.InterpModule
        self.interpolator = __import__('PYME.Analysis.FitFactories.Interpolators.' + interpModule , fromlist=['PYME', 'Analysis','FitFactories', 'Interpolators']).interpolator

        if 'Analysis.EstimatorModule' in metadata.getEntryNames():
            estimatorModule = metadata.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        self.startPosEstimator = __import__('PYME.Analysis.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'Analysis','FitFactories', 'zEstimators'])

        if fitfcn == f_Interp3d:
            if 'PSFFile' in metadata.getEntryNames():
                if self.interpolator.setModelFromMetadata(metadata):
                    print 'model changed'
                    self.startPosEstimator.splines.clear()

                if not 'z' in self.startPosEstimator.splines.keys():
                    self.startPosEstimator.calibrate(self.interpolator, metadata)
            else:
                self.interpolator.genTheoreticalModel(metadata)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        #setModel(md.PSFFile, md)
        interpolator = __import__('PYME.Analysis.FitFactories.Interpolators.' + md.Analysis.InterpModule , fromlist=['PYME', 'Analysis','FitFactories', 'Interpolators']).interpolator
        
        if 'Analysis.EstimatorModule' in md.getEntryNames():
            estimatorModule = md.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        #this is just here to make sure we clear our calibration when we change models        
        startPosEstimator = __import__('PYME.Analysis.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'Analysis','FitFactories', 'zEstimators'])        
        
        if interpolator.setModelFromFile(md.PSFFile, md):
            print 'model changed'
            startPosEstimator.splines.clear()

        X, Y, Z, safeRegion = interpolator.getCoords(md, slice(-roiHalfSize,roiHalfSize + 1), slice(-roiHalfSize,roiHalfSize + 1), slice(0,1))

        #X = 1e3*md.voxelsize.x*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        #Y = 1e3*md.voxelsize.y*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        #Z = array([0]).astype('f')

        return f_Interp3d(params, interpolator, X, Y, Z, safeRegion, md['PRI.Axis']), X.ravel()[0], Y.ravel()[0], Z.ravel()[0]

    def FromPoint(self, x, y, z=None, roiHalfSize=7, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()

        x0 = x
        y0 = y
        x = round(x)
        y = round(y)

        xslice = slice(max((x - roiHalfSize), 0),min((x + roiHalfSize + 1),self.data.shape[0]))
        yslice = slice(max((y - roiHalfSize), 0),min((y + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(0,1)
		
        
    #def __getitem__(self, key):
        #xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

        #generate grid to evaluate function on        
        X, Y, Z, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)


        #estimate some start parameters...

        if len(X.shape) > 1: #X is a matrix
            X_ = X[:, 0, 0]
            Y_ = Y[0, :, 0]
        else:
            X_ = X
            Y_ = Y

        


        #estimate errors in data
        nSlices = 1#dataROI.shape[2]

        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*dataROI)/self.metadata.Camera.ElectronsPerCount

        startParameters = self.startPosEstimator.getStartParameters(dataROI, X_, Y_) + [0.5,]

        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, self.interpolator, X, Y, Z, safeRegion, self.metadata['PRI.Axis'])

        fitErrors=None
        try:
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)

        #print res, fitErrors, resCode
        #return PSFFitResultR(res, self.metadata, numpy.array((sig_xl, sig_xr, sig_yu, sig_yd)),(xslice, yslice, zslice), resCode, fitErrors, numpy.array(startParameters), nchi2)
        return PSFFitResultR(res, self.metadata,(xslice, yslice, zslice), resCode, fitErrors, numpy.array(startParameters), nchi2)

     

#so that fit tasks know which class to use
FitFactory = PSFFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
