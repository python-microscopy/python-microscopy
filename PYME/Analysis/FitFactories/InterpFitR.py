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

#import scipy
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
#from pylab import *
import numpy as np
import types

from .fitCommon import fmtSlicesUsed 
from . import FFBase 

from PYME.Analysis._fithelpers import FitModelWeighted_, FitModelWeighted, FitModelWeightedJac

def f_Interp3d(p, interpolator, X, Y, Z, safeRegion, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p

    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0]), safeRegion[2][1])

    return interpolator.interp(X - x0 + 1, Y - y0 + 1, Z - z0 + 1)*A + b



fresultdtype=[('tIndex', '<i4'),
    ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]),
    ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]) ,
    #('coiR', [('sxl', '<f4'),('sxr', '<f4'),('syu', '<f4'),('syd', '<f4')]),
    ('resultCode', '<i4'),
    ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
    ('startParams', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]), ('nchi2', '<f4')]

def PSFFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, startParams=None, nchi2=-1):
	if fitErr == None:
		fitErr = -5e3*np.ones(fitResults.shape, 'f')

	if startParams == None:
		startParams = -5e3*np.ones(fitResults.shape, 'f')

	tIndex = metadata.tIndex

	return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, fmtSlicesUsed(slicesUsed), startParams.astype('f'), nchi2)], dtype=fresultdtype)


def genFitImage(fitResults, metadata, fitfcn=f_Interp3d):

    xslice = slice(*fitResults['slicesUsed']['x'])
    yslice = slice(*fitResults['slicesUsed']['y'])
    
    im = PSFFitFactory.evalModel(fitResults['fitResults'], metadata, np.mgrid[xslice].mean(), np.mgrid[yslice].mean())
    
    return im[0].squeeze()

def getDataErrors(im, metadata):
    dataROI = im - metadata.getEntry('Camera.ADOffset')

    return np.sqrt(metadata.getEntry('Camera.ReadNoise')**2 + (metadata.getEntry('Camera.NoiseFactor')**2)*metadata.getEntry('Camera.ElectronsPerCount')*metadata.getEntry('Camera.TrueEMGain')*dataROI)/metadata.getEntry('Camera.ElectronsPerCount')

		

class PSFFitFactory(FFBase.FFBase):
    def __init__(self, data, metadata, fitfcn=f_Interp3d, background=None, noiseSigma=None):
        super(PSFFitFactory, self).__init__(data, metadata, fitfcn, background, noiseSigma)
        
        #if type(fitfcn) == types.FunctionType: #single function provided - use numerically estimated jacobian
        #    self.solver = FitModelWeighted_
        #else: #should be a tuple containing the fit function and its jacobian
            
        
        if 'D' in dir(fitfcn):
            self.solver = FitModelWeightedJac
        else:
            self.solver = FitModelWeighted_
        

        interpModule = metadata.Analysis.InterpModule
        self.interpolator = __import__('PYME.Analysis.FitFactories.Interpolators.' + interpModule , fromlist=['PYME', 'Analysis','FitFactories', 'Interpolators']).interpolator

        if 'Analysis.EstimatorModule' in metadata.getEntryNames():
            estimatorModule = metadata.Analysis.EstimatorModule
        else:
            estimatorModule = 'astigEstimator'

        self.startPosEstimator = __import__('PYME.Analysis.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'Analysis','FitFactories', 'zEstimators'])

        if True:#fitfcn == f_Interp3d:
            if 'PSFFile' in metadata.getEntryNames():
                if self.interpolator.setModelFromMetadata(metadata):
                    print('model changed')
                    self.startPosEstimator.splines.clear()

                if not 'z' in self.startPosEstimator.splines.keys():
                    self.startPosEstimator.calibrate(self.interpolator, metadata)
            else:
                self.interpolator.genTheoreticalModel(metadata)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5, model=f_Interp3d):
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
            print('model changed')
            startPosEstimator.splines.clear()

        X, Y, Z, safeRegion = interpolator.getCoords(md, slice(x -roiHalfSize,x + roiHalfSize + 1), slice(y -roiHalfSize,y + roiHalfSize + 1), slice(0,1))

        return model(params, interpolator, X, Y, Z, safeRegion), X.ravel()[0], Y.ravel()[0], Z.ravel()[0]
        

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, dataMean, bgMean, sigma, xslice, yslice, zslice = self.getROIAtPoint(x,y,z,roiHalfSize, axialHalfSize)
        
        dataROI = dataMean - bgMean
        
        #generate grid to evaluate function on        
        X, Y, Z, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)
        
        if len(X.shape) > 1: #X is a matrix
            X_ = X[:, 0, 0]
            Y_ = Y[0, :, 0]
        else:
            X_ = X
            Y_ = Y

        #estimate start parameters        
        startParameters = self.startPosEstimator.getStartParameters(dataROI, X_, Y_)

        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, self.interpolator, X, Y, Z, safeRegion)

        fitErrors=None
        try:
            fitErrors = np.sqrt(np.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception:
            pass

        #normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)

        return PSFFitResultR(res, self.metadata,(xslice, yslice, zslice), resCode, fitErrors, np.array(startParameters), nchi2)

     

#so that fit tasks know which class to use
FitFactory = PSFFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.Analysis.MetaDataEdit as mde
from PYME.Analysis.FitFactories import Interpolators
from PYME.Analysis.FitFactories import zEstimators

#set of parameters that this fit needs to know about
PARAMETERS = [mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              mde.FilenameParam('PSFFile', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              #mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              #mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'none', choices=['x', 'y', 'none'])]
              
DESCRIPTION = '3D, single colour fitting using an interpolated measured PSF.'
LONG_DESCRIPTION = '3D, single colour fitting using an interpolated measured PSF. Should work for any 3D engineered PSF, with the default parameterisation optimised for astigmatism.'
USE_FOR = '3D single-colour'