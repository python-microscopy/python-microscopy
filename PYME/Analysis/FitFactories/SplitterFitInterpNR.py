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
#from pylab import *
#import copy_reg
import numpy
#import types

from . import InterpFitR
from .fitCommon import fmtSlicesUsed

from PYME.Analysis._fithelpers import *


def f_Interp3d2c(p, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, axialShift, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    Ag, Ar, x0, y0, z0, bG, bR = p

    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + axialShift), safeRegion[2][1] - axialShift)

    g = interpolator.interp(Xg - x0 + 1, Yg - y0 + 1, Zg - z0 + 1)*Ag + bG
    r = interpolator.interp(Xr - x0 + 1, Yr - y0 + 1, Zr - z0 + 1)*Ar + bR

    return numpy.concatenate((np.atleast_3d(g),np.atleast_3d(r)), 2)
    
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

        

#fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('bg', '<f4'), ('br', '<f4')]),
              ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('bg', '<f4'), ('br', '<f4')]),
              ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('bg', '<f4'), ('br', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('x2', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y2', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', [('g','<f4'),('r','<f4')]),
              ('nchi2', '<f4')]


def PSFFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=None, nchi2=-1, background=None):
    if fitErr == None:
        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')
        
    if background  == None:
        background = numpy.zeros(2, 'f')

    tIndex = metadata.tIndex
    return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), resultCode, fmtSlicesUsed(slicesUsed), background,nchi2)], dtype=fresultdtype) 
 
def BlankResult(metadata):
    r = numpy.zeros(1, fresultdtype)
    r['tIndex'] = metadata.tIndex
    r['fitError'].view('7f4')[:] = -5e3
    return r
		

def getDataErrors(im, metadata):
    dataROI = im - metadata.getEntry('Camera.ADOffset')

    return scipy.sqrt(metadata.getEntry('Camera.ReadNoise')**2 + (metadata.getEntry('Camera.NoiseFactor')**2)*metadata.getEntry('Camera.ElectronsPerCount')*metadata.getEntry('Camera.TrueEMGain')*dataROI)/metadata.getEntry('Camera.ElectronsPerCount')    



class InterpFitFactory(InterpFitR.PSFFitFactory):
    def __init__(self, data, metadata, fitfcn=f_Interp3d2c, background=None):
       super(InterpFitFactory, self).__init__(data, metadata, fitfcn, background) 
                
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
            print('model changed')
            startPosEstimator.splines.clear()

        Xg, Yg, Zg, safeRegion = interpolator.getCoords(md, slice(-roiHalfSize,roiHalfSize + 1), slice(-roiHalfSize,roiHalfSize + 1), slice(0,1))
        
        DeltaX = md.chroma.dx.ev(x, y)
        DeltaY = md.chroma.dy.ev(x, y)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY
        Zr = Zg + md.Analysis.AxialShift

        return f_Interp3d2c(params, interpolator, Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, md.Analysis.AxialShift), Xg.ravel()[0], Yg.ravel()[0], Zg.ravel()[0]
		
        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2 = self.getSplitROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)
        
        if min(dataROI.shape[:2]) < 4: # too small to fit
            return BlankResult(self.metadata)
      
        
        dataROI = np.maximum(dataROI - bgROI, -sigma)
        
        dx_ = Xg[0] - Xr[0]
        dy_ = Yg[0] - Yr[0]
        
        zslice = slice(0,2)        
        Xg, Yg, Zg, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)        

        if len(Xg.shape) > 1: #X is a matrix
            X_ = Xg[:, 0, 0]
            Y_ = Yg[0, :, 0]
        else:
            X_ = Xg
            Y_ = Yg

        Xr = Xg - dx_
        Yr = Yg - dy_
        Zr = Zg + self.metadata.Analysis.AxialShift
                

        #estimate some start parameters...
        Ag = dataROI[:,:,0].max() - dataROI[:,:,0].min() #amplitude
        Ar = dataROI[:,:,1].max() - dataROI[:,:,1].min() #amplitude        
        
        
        z0 = 0        
        if 'TWOCHANNEL' in dir(self.startPosEstimator):
            startParams = self.startPosEstimator.getStartParameters(dataROI, X_, Y_)
        
        else:
            if Ag > Ar: #use prightest channel for start parameter estimation
                startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,:1], X_, Y_)
            else:
                startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,1:], X_, Y_)
                z0 = self.metadata.Analysis.AxialShift

        startParameters = [Ag*startParams[0]/(Ag + Ar), Ar*startParams[0]/(Ag + Ar), startParams[1], startParams[2], z0 + startParams[3], dataROI[:,:,0].min(),dataROI[:,:,1].min()]
        #print startParameters

        #print dataROI.shape
	
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, self.interpolator,Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, self.metadata.Analysis.AxialShift)

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception:
            pass

        #normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)

	#print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, np.array(startParameters), (xslice, yslice, xslice2, yslice2), resCode, fitErrors, nchi2, bgROI.mean(0).mean(0))
        #return PSFFitResultR(res, self.metadata, , resCode, fitErrors, numpy.array(startParameters), nchi2)
    
        

#so that fit tasks know which class to use
FitFactory = InterpFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.Analysis.MetaDataEdit as mde
from PYME.Analysis.FitFactories import Interpolators
from PYME.Analysis.FitFactories import zEstimators

PARAMETERS = [mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              mde.FilenameParam('PSFFile', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'none', choices=['x', 'y', 'none']),
              ]
              
DESCRIPTION = 'Ratiometric multi-colour 3D PSF fit (large shifts).'
LONG_DESCRIPTION = 'Ratiometric multi-colour 3D PSF fit (large shifts). The 3D version of SplitterFitFNR'