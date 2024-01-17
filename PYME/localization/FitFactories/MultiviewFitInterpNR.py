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
import scipy.ndimage as ndimage
#from pylab import *
#import copy_reg
import numpy
#import types

from . import InterpFitR
from .fitCommon import fmtSlicesUsed

from PYME.Analysis._fithelpers import *



def f_Interp3Dmultiview(p, interpolator, Xvs, Yvs, Zvs, safeRegion, axialShifts, nViews, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    
    #This bit of the parameter vector is the position
    x0, y0, z0 = p[:3]
    
    #next come the bits which can vary with the number of views
    p_var = p[3:]
    
    #we always have amplitudes for each of the views
    As = p_var[:nViews]
    
    dxs = 0*As
    dys = 0*As
    
    if len(p_var) > nViews:
        #we can also have backgrounds for each view
        Bgs = p_var[nViews:(2*nViews)]
        
        if len(p_var) > (2*nViews):
            #and shifts for all but the first view
            pr = p_var[(2*nViews):]
            dxs[1:] = pr[:(nViews -1)]
            dys[1:] = pr[(nViews - 1):]
    else:
        Bgs = 0*As


    #currently just come to a hard stop when the optimiser tries to leave the safe region
    #prob. not ideal, for a number of reasons
    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + np.max(axialShifts)), safeRegion[2][1] - np.min(axialShifts))

    views = np.zeros([len(Xvs[0]), len(Yvs[0]), nViews])
    
    for i in range(nViews):
        views[:,:,i] = interpolator.interp(Xvs[i] - x0 + 1 + dxs[i], Yvs[i] - y0 + 1 + dys[i], Zvs[i] - z0 + 1)*As[i] + Bgs[i]

    return views
    
def f_J_Interp3Dmultiview(p,interpolator, Xvs, Yvs, Zvs, safeRegion, axialShifts, nViews, *args):
    """generate the jacobian - for use with _fithelpers.weightedJacF"""

    #This bit of the parameter vector is the position
    x0, y0, z0 = p[:3]

    #next come the bits which can vary with the number of views
    p_var = p[3:]

    #we always have amplitudes for each of the views
    As = p_var[:nViews]

    dxs = 0 * As
    dys = 0 * As

    if len(p_var) > nViews:
        #we can also have backgrounds for each view
        Bgs = p_var[nViews:(2 * nViews)]
    
        if len(p_var) > (2 * nViews):
            #and shifts for all but the first view
            pr = p_var[(2 * nViews):]
            dxs[1:] = pr[:(nViews - 1)]
            dys[1:] = pr[(nViews - 1):]
    else:
        Bgs = 0 * As

    x0 = min(max(x0, safeRegion[0][0]), safeRegion[0][1])
    y0 = min(max(y0, safeRegion[1][0]), safeRegion[1][1])
    z0 = min(max(z0, safeRegion[2][0] + np.max(axialShifts)), safeRegion[2][1] - np.min(axialShifts))

    dX = np.zeros([len(Xvs[0]), len(Yvs[0]), nViews])
    dY = np.zeros_like(dX)
    dZ = np.zeros_like(dX)
    
    dAs = []
    dBgs = []

    for i in range(nViews):
        view_i = interpolator.interp(Xvs[i] - x0 + 1 + dxs[i], Yvs[i] - y0 + 1 + dys[i], Zvs[i] - z0 + 1) * As[i] + Bgs[i]
        gx_, gy_, gz_ = interpolator.interpG(Xvs[i] - x0 + 1 + dxs[i], Yvs[i] - y0 + 1 + dys[i], Zvs[i] - z0 + 1)
        
        dX[:,:,i] = As[i]*gx_
        dY[:, :, i] = As[i] * gy_
        dZ[:, :, i] = As[i] * gz_
    
        dAi = np.zeros_like(dX)
        dAi[:,:,i] = view_i
        dAs.append(dAi.ravel()[:,None])
        
        dBi = np.zeros_like(dX)
        dBi[:,:,i] = 1
        dBgs.append(dBi.ravel()[:,None])
    
    return numpy.hstack([dX.ravel()[:,None], dY.ravel()[:,None], dZ.ravel()[:,None]] + dAs + dBgs)

        

#fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

# fresultdtype=[('tIndex', '<i4'),
#               ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')]),
#               ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')]),
#               ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')]),
#               ('resultCode', '<i4'),
#               ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
#                               ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
#                               ('x2', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
#                               ('y2', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
#               ('subtractedBackground', [('g','<f4'),('r','<f4')]),
#               ('nchi2', '<f4')]


def fresult_dtype(nViews=4):
    r_dt = [('x0', '<f4'),('y0', '<f4'),('z0', '<f4')] + [('A%d' % i, 'f4') for i in range(nViews)] + \
            [('b%d' % i, 'f4') for i in range(nViews)] + \
           [('dx%d' % i, 'f4') for i in range(nViews)] + [('dy%d' % i, 'f4') for i in range(nViews)]
    
    dt = [('tIndex', '<i4'),
          ('fitResults', r_dt),
          ('fitError', r_dt),
          ('startParams', r_dt),
          ('resultCode', '<i4'),
          ('slicesUsed', [('x%d', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]) for i in range(nViews)] + \
                          [('y%d', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]) for i in range(nViews)]),
          ('subtractedBackground', [('v%d','<f4') for i in range(nViews)]),
          ('nchi2', '<f4')]

#
#def PSFFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=None, nchi2=-1, background=None):
#    if fitErr == None:
#        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')
#        
#    if background  == None:
#        background = numpy.zeros(2, 'f')
#
#    tIndex = metadata.tIndex
#    return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), resultCode, fmtSlicesUsed(slicesUsed), background,nchi2)], dtype=fresultdtype) 
    
def PSFFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=-5e3, nchi2=-1, background=0, nViews=4, dt=fresult_dtype(4)):
    fr = np.zeros(1, dtype=dt)
    
    flat_view = '%df4' % len(dt['fitResults'])
    
    n = len(fitResults)

    fr['tIndex'] = metadata['tIndex']
    fr['resultCode'] = resultCode
    fr['nchi2'] = nchi2
    #print n, fr['fitResults'].view('f4').shape
    fr['fitResults'].view(flat_view)[0,:n] = fitResults
    fr['startParams'].view(flat_view)[0,:n] = startParams
    
    if fitErr is None:
        fr['fitError'].view(flat_view)[0,:] = -5e3
    else:
        fr['fitError'].view(flat_view)[0,:n] = fitErr
        
    fr['subtractedBackground'].view('%if4' % nViews)[:] = background
    slu = np.array(fmtSlicesUsed(slicesUsed), dtype='i4')
    #print slu.shape, fr['slicesUsed'].view('12i4').shape, slu.dtype, slu.ravel().shape
    fr['slicesUsed'].view('%ii4' % (6*nViews))[:] = slu.ravel()
        
    return fr 
 
def BlankResult(metadata, dt):
    r = numpy.zeros(1, dt)
    r['tIndex'] = metadata['tIndex']
    r['fitError'].view('f4')[:] = -5e3
    return r


def getDataErrors(im, metadata):
    # TODO - Fix me for camera maps (ie use correctImage function not ADOffset) or remove
    dataROI = im - metadata.getEntry('Camera.ADOffset')

    return scipy.sqrt(metadata.getEntry('Camera.ReadNoise')**2 + (metadata.getEntry('Camera.NoiseFactor')**2)*metadata.getEntry('Camera.ElectronsPerCount')*metadata.getEntry('Camera.TrueEMGain')*dataROI)/metadata.getEntry('Camera.ElectronsPerCount')    

def genFitImage(fitResults, metadata):
    from PYME.IO.MetaDataHandler import get_camera_roi_origin
    xslice = slice(*fitResults['slicesUsed']['x'])
    yslice = slice(*fitResults['slicesUsed']['y'])

    vx, vy, _ = metadata.voxelsize_nm
    
    #position in nm from camera origin
    roi_x0, roi_y0 = get_camera_roi_origin(metadata)

    x_ = (xslice.start + roi_x0)*vx
    y_ = (yslice.start + roi_y0)*vy
    
    #ratio = fitResults['ratio']
    
    im = InterpFitFactory._evalModel(np.array(list(fitResults['fitResults'])), metadata, xslice, yslice, x_, y_)[0]
    #print im.shape

    return np.hstack([im[:,:,0], im[:,:,1]]).squeeze()

class InterpFitFactory(InterpFitR.PSFFitFactory):
    def __init__(self, data, metadata, fitfcn=f_Interp3Dmultiview, background=None, noiseSigma=None, nViews=4, **kwargs):
        super(InterpFitFactory, self).__init__(data, metadata, fitfcn, background, noiseSigma, **kwargs)
        
        self.nViews = nViews
        self.rdtype = fresult_dtype(nViews)
                
    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        xs = slice(-roiHalfSize,roiHalfSize + 1)
        ys = slice(-roiHalfSize,roiHalfSize + 1)

        return cls._evalModel(params, md, xs, ys, x, y)        
        
    @classmethod    
    def _evalModel(cls, params, md, xs, ys, x, y, nViews=4):
        #generate grid to evaluate function on
        #setModel(md.PSFFile, md)
        interpolator = __import__('PYME.localization.FitFactories.Interpolators.' + md.getOrDefault('Analysis.InterpModule', 'CSInterpolator') , fromlist=['PYME', 'localization', 'FitFactories', 'Interpolators']).interpolator
        
        if 'Analysis.EstimatorModule' in md.getEntryNames():
            estimatorModule = md['Analysis.EstimatorModule']
        else:
            estimatorModule = 'astigEstimator'

        #this is just here to make sure we clear our calibration when we change models        
        startPosEstimator = __import__('PYME.localization.FitFactories.zEstimators.' + estimatorModule , fromlist=['PYME', 'localization', 'FitFactories', 'zEstimators'])        
        
        if interpolator.setModelFromFile(md['PSFFile'], md):
            print('model changed')
            startPosEstimator.splines.clear()

        X, Y, Z, safeRegion = interpolator.getCoords(md, xs, ys, slice(0,1))

        vx, vy, _ = md.voxelsize_nm
        
        DeltaX = md['chroma.dx'].ev(x, y)
        DeltaY = md['chroma.dy'].ev(x, y)

        dxp = int(DeltaX/vx)
        dyp = int(DeltaY/vy)

        Xr = Xg + DeltaX - vx*dxp
        Yr = Yg + DeltaY - vx*dyp
        Zr = Zg + md['Analysis.AxialShifts']

        return f_Interp3Dmultiview(params, interpolator, Xvs, Yvs, Zvs, safeRegion, md['Analysis.AxialShifts'], nViews), X.ravel()[0], Y.ravel()[0], Z.ravel()[0]

        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        #Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2 = self.getSplitROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        Xs, Ys, dataROI, bgROI, sigma, xslices, yslices = self.getMultiviewROIAtPoint(x, y, z,roiHalfSize,axialHalfSize)
        
        if min(dataROI.shape[:2]) < 4: # too small to fit
            return BlankResult(self.metadata)
            
        
        dataROI = dataROI-bgROI
        
        dxs_ = [Xs[0][0] - Xs[i][0] for i in range(self.nViews)]
        dys_ = [Ys[0][0] - Ys[i][0] for i in range(self.nViews)]
        
        zslice = slice(0,2)        
        X, Y, Z, safeRegion = self.interpolator.getCoords(self.metadata, xslices[0], yslices[0], zslice)

        if len(X.shape) > 1: #X is a matrix
            X_ = X[:, 0, 0]
            Y_ = Y[0, :, 0]
        else:
            X_ = X
            Y_ = Y

        Xvs = [X - dxs_[i] for i in range(self.nViews)]
        Yvs = [Y - dys_[i] for i in range(self.nViews)]
        Zvs = [Z + self.metadata['Analysis.AxialShifts'][i] for i in range(self.nViews)]
                

        #estimate some start parameters...
        As = [dataROI[:,:,i].max() - dataROI[:,:,i].min() for i in range(self.nViews)] #amplitude
        
        z0 = 0        
        if 'MULTICHANNEL' in dir(self.startPosEstimator):
            startParams = self.startPosEstimator.getStartParameters(dataROI, X_, Y_)
            
            spA = As*startParams[0]/np.sum(As)
        else:
            brightest = np.argmax(As)
            startParams = self.startPosEstimator.getStartParameters(dataROI[:,:,brightest:(brightest+1)], X_, Y_)
            z0 = self.metadata['Analysis.AxialShift']
            spA = As * startParams[0]/(As[brightest])
            

        fitBackground = self.metadata.getOrDefault('Analysis.FitBackground', True)
        fitShifts = self.metadata.getOrDefault('Analysis.FitShifts', False)

        if fitBackground:
            sbg = [dataROI[:,:,i].min() for i in range(self.nViews)]
            if fitShifts:
                startParameters = [startParams[1], startParams[2], z0 + startParams[3]] + spA + sbg + list(np.zeros(self.nViews -1))
            else:
                startParameters = [startParams[1], startParams[2], z0 + startParams[3]] + spA + sbg
                
        else:
            startParameters = [startParams[1], startParams[2], z0 + startParams[3]] + spA
        #print startParameters
        
        startParameters = np.array(startParameters)
        
 
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        if  self.metadata.getOrDefault('Analysis.PoissonML', False):
            res = FitModelPoissonBFGS(self.fitfcn, startParameters, dataROI + bgROI, bgROI, self.interpolator,Xvs, Yvs, Zvs, safeRegion, self.metadata['Analysis.AxialShifts'], self.nViews)[0]
            cov_x = np.eye(len(res))            
            infodict = {'fvec': self.fitfcn(res, self.interpolator,Xvs, Yvs, Zvs, safeRegion, self.metadata['Analysis.AxialShifts'], self.nViews) - (dataROI + bgROI)}
            resCode = 1
        else:
            (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, self.interpolator,Xvs, Yvs, Zvs, safeRegion, self.metadata['Analysis.AxialShifts'], self.nViews)
            #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted_D(self.fitfcn, startParameters, dataROI, sigma, pScale, self.interpolator,Xg, Yg, Zg, Xr, Yr, Zr, safeRegion, self.metadata.Analysis.AxialShift)

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x) * (infodict['fvec'] * infodict['fvec']).sum() / (len(dataROI.ravel())- len(res)))
        except Exception:
            pass
        
        #print infodict['nfev']

        #normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)

        #print res, fitErrors, resCode
        return PSFFitResultR(res, self.metadata, np.array(startParameters), (xslices, yslices), resCode, fitErrors,
                             nchi2, bgROI.mean(0).mean(0), nViews=self.nViews, dt=self.rdtype)
        #return PSFFitResultR(res, self.metadata, , resCode, fitErrors, numpy.array(startParameters), nchi2)
    
        

#so that fit tasks know which class to use
FitFactory = InterpFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresult_dtype #only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde
from PYME.localization.FitFactories import Interpolators
from PYME.localization.FitFactories import zEstimators

PARAMETERS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','CSInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              mde.FilenameParam('PSFFile', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf|TIFF files|*.tif'),
              mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', -270),
              mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'none', choices=['x', 'y', 'none']),
              mde.BoolParam('Analysis.FitBackground', 'Fit Background', True),
              #mde.FloatListParam('chroma.ChannelRatios', 'Channel Ratios', [0.7]),
              ]
              
DESCRIPTION = 'Ratiometric multi-colour 3D PSF fit (large shifts).'
LONG_DESCRIPTION = 'Ratiometric multi-colour 3D PSF fit (large shifts). The 3D version of SplitterFitFNR'
