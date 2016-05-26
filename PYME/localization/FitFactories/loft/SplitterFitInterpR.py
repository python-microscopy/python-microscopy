#!/usr/bin/python

##################
# LatGaussFitFRTC.py
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
#from PYME.localization.FitFactories.zEstimators import astigEstimator

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

def f_Interp3d2c(p, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    Ag, Ar, x0, y0, z0, bG, bR = p

    #make sure our model is big enough to stretch to our current position
#    xm = len(Xg)/2
#    dx = min((interpolator.shape[0] - len(Xg))/2, xm) - 2
#
#    ym = len(Yg)/2
#    dy = min((interpolator.shape[1] - len(Yg))/2, ym) - 2
#
#    x0 = min(max(x0, Xg[xm - dx]), Xg[dx + xm])
#    y0 = min(max(y0, Yg[ym - dy]), Yg[dy + ym])
#    z0 = min(max(z0, max(Zg[0], Zr[0]) + interpolator.IntZVals[2]), min(Zg[0], Zr[0]) + interpolator.IntZVals[-2])

    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + axialShift), safeRegion[2][1] - axialShift)

    g = interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg[0] - z0 + 1)*Ag + bG
    r = interpolator.interp(Xr - x0 + 1, Yr - y0 + 1, Zr[0] - z0 + 1)*Ar + bR

    return numpy.concatenate((g,r), 2)
    
def f_J_Interp3d2c(p,interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift, *args):
    '''generate the jacobian - for use with _fithelpers.weightedJacF'''
    Ag, Ar, x0, y0, z0, bG, bR = p

    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + axialShift), safeRegion[2][1] - axialShift)

    g = interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg[0] - z0 + 1)
    r = interpolator.interp(Xr - x0 + 1, Yr - y0 + 1, Zr[0] - z0 + 1)
    
    gx, gy, gz = interpolator.interpG(Xg - x0 + 1, Yg - y0 + 1, Zg[0] - z0 + 1)
    rx, ry, rz = interpolator.interpG(Xr - x0 + 1, Yr - y0 + 1, Zr[0] - z0 + 1)
    
    bg = np.ones_like(gx)
    zb = np.zeros_like(gx)
    
    dAg = numpy.concatenate((g,zb), 2).ravel()[:,None]
    dAr = numpy.concatenate((zb,r), 2).ravel()[:,None]
    dX = numpy.concatenate((Ag*gx,Ar*rx), 2).ravel()[:,None]
    dY = numpy.concatenate((Ag*gy,Ar*ry), 2).ravel()[:,None]
    dZ = numpy.concatenate((Ag*gz,Ar*rz), 2).ravel()[:,None]
    dBg = numpy.concatenate((bg,zb), 2).ravel()[:,None]
    dBr = numpy.concatenate((zb,bg), 2).ravel()[:,None]
    
    #r = r.reshape((-1, 7))
    return numpy.hstack([dAg, dAr, dX, dY, dZ, dBg, dBr])
    
#f_Interp3d2c.D = f_J_Interp3d2c

def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n



fresultdtype=[('tIndex', '<i4'),
    ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4')]),
    ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4')]),
    ('resultCode', '<i4'), 
    ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
    ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('backgroundG', '<f4'), ('backgroundR', '<f4')]),
    ('nchi2', '<f4')]

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

		

def getDataErrors(im, metadata):
    dataROI = im - metadata.getEntry('Camera.ADOffset')

    return scipy.sqrt(metadata.getEntry('Camera.ReadNoise')**2 + (metadata.getEntry('Camera.NoiseFactor')**2)*metadata.getEntry('Camera.ElectronsPerCount')*metadata.getEntry('Camera.TrueEMGain')*dataROI)/metadata.getEntry('Camera.ElectronsPerCount')



class PSFFitFactory:
    def __init__(self, data, metadata, fitfcn=f_Interp3d2c, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in 
        metadata. '''
        self.data = data
        self.metadata = metadata
        self.background = background
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if not 'D' in dir(fitfcn): #single function provided - use numerically estimated jacobian
            self.solver = FitModelWeighted_
        else: #should be a tuple containing the fit function and its jacobian
            self.solver = FitModelWeightedJac_


        interpModule = metadata.Analysis.InterpModule
        self.interpolator = __import__('PYME.localization.FitFactories.Interpolators.' + interpModule , fromlist=['PYME', 'localization', 'FitFactories', 'Interpolators']).interpolator

        if 'Analysis.EstimatorModule' in metadata.getEntryNames():
            estimatorModule = metadata.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        self.startPosEstimator = __import__('PYME.localization.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'localization', 'FitFactories', 'zEstimators'])

        if fitfcn == f_Interp3d2c:
            if 'PSFFile' in metadata.getEntryNames():
                if self.interpolator.setModelFromMetadata(metadata):
                    print('model changed')
                    self.startPosEstimator.splines.clear()

                if not 'z' in self.startPosEstimator.splines.keys():
                    self.startPosEstimator.calibrate(self.interpolator, metadata)
            else:
                self.interpolator.genTheoreticalModel(metadata)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        #setModel(md.PSFFile, md)
        interpolator = __import__('PYME.localization.FitFactories.Interpolators.' + md.Analysis.InterpModule , fromlist=['PYME', 'localization', 'FitFactories', 'Interpolators']).interpolator
        
        if 'Analysis.EstimatorModule' in md.getEntryNames():
            estimatorModule = md.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        #this is just here to make sure we clear our calibration when we change models        
        startPosEstimator = __import__('PYME.localization.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'localization', 'FitFactories', 'zEstimators'])        
        
        if interpolator.setModelFromFile(md.PSFFile, md):
            print('model changed')
            startPosEstimator.splines.clear()

#        if 'Analysis.EstimatorModule' in md.getEntryNames():
#            estimatorModule = metadata.Analysis.EstimatorModule
#        else:
#            estimatorModule = 'astigEstimator'
#
#        startPosEstimator = __import__('PYME.localization.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'localization', 'FitFactories', 'zEstimators'])
#
#        startPosEstimator.calibrate(interpolator, md)

        Xg, Yg, Zg, safeRegion = interpolator.getCoords(md, slice(-roiHalfSize,roiHalfSize + 1), slice(-roiHalfSize,roiHalfSize + 1), slice(0,1))
        
        DeltaX = md.chroma.dx.ev(x, y)
        DeltaY = md.chroma.dy.ev(x, y)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY
        Zr = Zg + md.Analysis.AxialShift

        #X = 1e3*md.voxelsize.x*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        #Y = 1e3*md.voxelsize.y*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        #Z = array([0]).astype('f')

        return f_Interp3d2c(params, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, md.Analysis.AxialShift), Xg.ravel()[0], Yg.ravel()[0], Zg.ravel()[0]

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()

        x0 = x
        y0 = y
        x = round(x)
        y = round(y)

        xslice = slice(max((x - roiHalfSize), 0),min((x + roiHalfSize + 1),self.data.shape[0]))
        yslice = slice(max((y - roiHalfSize), 0),min((y + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(0,2)
        

    #def __getitem__(self, key):
        #xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice] - self.metadata.Camera.ADOffset
        #print self.data.shape, dataROI.shape

        #generate grid to evaluate function on        
        Xg, Yg, Zg, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)


        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)
        x_ = Xg.mean() + (self.metadata.Camera.ROIPosX - 1)*1e3*self.metadata.voxelsize.x
        y_ = Yg.mean() + (self.metadata.Camera.ROIPosY - 1)*1e3*self.metadata.voxelsize.y
        DeltaX = self.metadata.chroma.dx.ev(x_, y_)
        DeltaY = self.metadata.chroma.dy.ev(x_, y_)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY
        Zr = Zg + self.metadata.Analysis.AxialShift

        #print DeltaX
        #print DeltaY

        

        if len(Xg.shape) > 1: #X is a matrix
            X_ = Xg[:, 0, 0]
            Y_ = Yg[0, :, 0]
        else:
            X_ = Xg
            Y_ = Yg

        #x0 =  Xg.mean()
        #y0 =  Yg.mean()
        #z0 = 200.0

        

        #startParameters = [Ag, Ar, x0, y0, 250/2.35, dataROI[:,:,0].min(),dataROI[:,:,1].min(), .001, .001]

	
        #estimate errors in data
        nSlices = 1#dataROI.shape[2]
        
        #sigma = scipy.sqrt(self.metadata.CCD.ReadNoise**2 + (self.metadata.CCD.noiseFactor**2)*self.metadata.CCD.electronsPerCount*self.metadata.CCD.EMGain*dataROI)/self.metadata.CCD.electronsPerCount
        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*scipy.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount + 1


        if not self.background == None and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            #print 'bgs'
            if len(numpy.shape(self.background)) > 1:
                bgROI = self.background[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

                dataROI = dataROI - bgROI
            else:
                dataROI = dataROI - (self.background - self.metadata.Camera.ADOffset)

        #estimate some start parameters...
        Ag = dataROI[:,:,0].max() - dataROI[:,:,0].min() #amplitude
        Ar = dataROI[:,:,1].max() - dataROI[:,:,1].min() #amplitude        
        
        z0 = 0
        if Ag > Ar: #use prightest channel for start parameter estimation
            startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,:1], X_, Y_)
        else:
            startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,1:], X_, Y_)
            z0 = self.metadata.Analysis.AxialShift

        startParameters = [Ag*startParams[0]/(Ag + Ar), Ar*startParams[0]/(Ag + Ar), startParams[1], startParams[2], z0 + startParams[3], dataROI[:,:,0].min(),dataROI[:,:,1].min()]

        #print dataROI.shape
	
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, self.interpolator,Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, np.abs(self.metadata.Analysis.AxialShift))

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

        #normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)

	#print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, numpy.array(startParameters), nchi2)

   
        
#so that fit tasks know which class to use
FitFactory = PSFFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde
from PYME.localization.FitFactories import Interpolators
from PYME.localization.FitFactories import zEstimators

PARAMETERS = [mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              mde.FilenameParam('PSFFile', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'none', choices=['x', 'y', 'none']),
              ]
              
DESCRIPTION = 'Ratiometric multi-colour 3D PSF fit.'
LONG_DESCRIPTION = 'Ratiometric multi-colour 3D PSF fit. The 3D version of SplitterFitFR'
