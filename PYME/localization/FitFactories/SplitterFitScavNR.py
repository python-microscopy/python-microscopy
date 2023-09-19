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
import numpy


#import PYME.Analysis.points.twoColour as twoColour
from .fitCommon import fmtSlicesUsed 
from . import FFBase 

from PYME.localization.cModels.gauss_app import *
from PYME.Analysis._fithelpers import *

from .LatGaussFitFR import f_gauss2d, f_gauss2d_no_bg

def f_gauss2d2c(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR, b_x, b_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    #g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....
    
    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)

def f_gauss2d2cA(p, Xg, Yg, Xr, Yr, Arr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    #Ag,Ar, x0, y0, s, bG, bR, b_x, b_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    #genGaussInArray(Arr[:,:,0], Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    #genGaussInArray(Arr[:,:,1],Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    #genSplitGaussInArray(Arr,Xg, Yg,Xr,Yr,Ag, Ar,x0,y0,s,bG,b_x,b_y)
    genSplitGaussInArrayPVec(p,Xg, Yg,Xr,Yr,Arr)
    #g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....

    return Arr #numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)

def f_gauss2d2ccb(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR = p)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG = p)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....

    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)




#fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('bg', '<f4'), ('br', '<f4')]),
              ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('bg', '<f4'), ('br', '<f4')]),
              ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('bg', '<f4'), ('br', '<f4')]), 
              ('subtractedBackground', [('g','<f4'),('r','<f4')]),
              ('sumIntensity', [('g','<f4'),('r','<f4')]),
              ('nchi2', '<f4'),
              ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]


#def GaussianFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=None, background = None):
#    if fitErr == None:
#        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')
#
#    if background  == None:
#        background = numpy.zeros(2, 'f')
#
#    tIndex = metadata.tIndex
#
#    return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), background, resultCode, fmtSlicesUsed(slicesUsed))], dtype=fresultdtype) 

def GaussianFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=-5e3, nchi2=-1, background=0, sumIntensity = 0):
    fr = np.zeros(1, dtype=fresultdtype)
    
    n = len(fitResults)

    fr['tIndex'] = metadata['tIndex']
    fr['resultCode'] = resultCode
    fr['nchi2'] = nchi2
    #print n, fr['fitResults'].view('f4').shape
    fr['fitResults'].view('7f4')[:n] = fitResults
    fr['startParams'].view('7f4')[:n] = startParams
    
    if fitErr is None:
        fr['fitError'].view('7f4')[:] = -5e3
    else:
        fr['fitError'].view('7f4')[:n] = fitErr
        
    fr['subtractedBackground'].view('2f4')[:] = background
    fr['sumIntensity'].view('2f4')[:] = sumIntensity
    slu = np.array(fmtSlicesUsed(slicesUsed), dtype='i4')
    #print slu.shape, fr['slicesUsed'].view('12i4').shape, slu.dtype, slu.ravel().shape
    fr['slicesUsed'].view('6i4')[:] = slu.ravel()
        
    return fr 
 
def BlankResult(metadata):
    r = numpy.zeros(1, fresultdtype)
    r['tIndex'] = metadata['tIndex']
    r['fitError'].view('7f')[:] = -5e3
    return r
		
def splWrap(*args):
    #print ''
    #args = args[:]
    return splitGaussWeightedMisfit(*args).copy()
    
    
def genFitImage(fitResults, metadata):
    #if fitfcn == f_Interp3d:
    #    if 'PSFFile' in metadata.getEntryNames():
    #        setModel(metadata.getEntry('PSFFile'), metadata)
    #    else:
    #        genTheoreticalModel(metadata)

    xslice = slice(*fitResults['slicesUsed']['x'])
    yslice = slice(*fitResults['slicesUsed']['y'])
    
    vx, vy, _ = metadata.voxelsize_nm
    
    #position in nm from camera origin
    roi_x0, roi_y0 = FFBase.get_camera_roi_origin(metadata)

    x_ = (xslice.start + roi_x0) * vx
    y_ = (yslice.start + roi_y0) * vy
    
    #look up shifts
    DeltaX = metadata['chroma.dx'].ev(x_, y_)
    DeltaY = metadata['chroma.dy'].ev(x_, y_)
    
    dxp = int(DeltaX/vx)
    dyp = int(DeltaY/vy)
    
    Xg = vx*scipy.mgrid[xslice]
    Yg = vy*scipy.mgrid[yslice]

    #generate a corrected grid for the red channel
    #note that we're cheating a little here - for shifts which are slowly
    #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
    #similarly for y. For slowly varying shifts the following should be
    #equivalent to this. For rapidly varying shifts all bets are off ...

    Xr = Xg + DeltaX - vx*dxp
    Yr = Yg + DeltaY - vy*dyp

    #X = 1e3*metadata.getEntry('voxelsize.x')*scipy.mgrid[xslice]
    #Y = 1e3*metadata.getEntry('voxelsize.y')*scipy.mgrid[yslice]
    #Z = array([0]).astype('f')
    #P = scipy.arange(0,1.01,.01)

    #im = fitfcn(fitResults['fitResults'], X, Y, Z, P).reshape(len(X), len(Y))
    #buf = numpy.zeros()
    d = np.zeros([Xr.shape[0], Yr.shape[0], 2], order='F')
    s = np.ones_like(d)
    buf = np.zeros(d.size)
    print((d.shape, Xr.shape, Yr.shape))
    im = -splWrap(np.array(list(fitResults['fitResults'])), d, s, Xg, Yg, Xr, Yr, buf).reshape(d.shape, order='F')

    return np.hstack([im[:,:,0], im[:,:,1]]).squeeze()
    
def _munge_chans(res, A, bg, brightest_chan):
    # munge results - TODO make this a bit cleaner / faster
    r2 = np.zeros(7)
    r2[:len(res)] = res
    res = np.zeros(7)
    
    if brightest_chan == 0:
        res[0] = r2[0]
        res[1] = A*r2[0]
        res[5] = r2[4]
        res[6] = bg
    else:
        res[1] = r2[0]
        res[0] = A*r2[0]
        res[6] = r2[5]
        res[5] = bg
    
    res[2:5] = r2[1:4]
    
    return res

class GaussianFitFactory(FFBase.FFBase):
    def __init__(self, data, metadata, fitfcn=genSplitGaussInArrayPVec, background=None, noiseSigma=None, **kwargs):
        super(GaussianFitFactory, self).__init__(data, metadata, fitfcn, background, noiseSigma, **kwargs)
        
        #if False:#'D' in dir(fitfcn): #function has jacobian
        #    self.solver = FitModelWeightedJac
        #else:
        #    self.solver = FitModelWeighted
    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        Xg = x + vs.x*scipy.mgrid[slice(-roiHalfSize,roiHalfSize + 1)]
        Yg = y + vs.y*scipy.mgrid[slice(-roiHalfSize,roiHalfSize + 1)]

        #generate a corrected grid for the red channel      
        DeltaX = md['chroma.dx'].ev(x, y)
        DeltaY = md['chroma.dy'].ev(x, y)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY
        
        #print DeltaX, DeltaY
        d = np.zeros([Xr.shape[0], Yr.shape[0], 2], order='F')
        s = np.ones_like(d)
        buf = np.zeros(d.size)
        #print((d.shape, Xr.shape, Yr.shape))
        im = -splWrap(np.array(params), d, s, Xg, Yg, Xr, Yr, buf).reshape(d.shape, order='F')
    
        #return np.concatenate([im[:,:,0], im[:,:,1]], 2).squeeze(), Xg.ravel()[0], Yg.ravel()[0], 0
        return im, Xg.ravel()[0], Yg.ravel()[0], 0


        #return splWrap(params, Xg, Yg, Xr, Yr), Xg.ravel()[0], Yg.ravel()[0], 0
		
        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2 = self.getSplitROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)
        
        if min(dataROI.shape[:2]) < 4: # too small to fit
            return BlankResult(self.metadata)
        
        dataROI = np.maximum(dataROI - bgROI, -sigma)
        sI = dataROI.mean(0).mean(0)
        
        if bgROI.ndim == 3:
            bgs = bgROI.mean(0).mean(0)
        else:
            bgs = bgROI
        
        brightest_chan = np.argmax(sI) #only perform fit on brightest channel
        
        chandata = dataROI[:,:,brightest_chan]
        chansigma = sigma[:,:,brightest_chan]
        
        #estimate some start parameters...
        A = chandata.max() - chandata.min() #amplitude
        
        if brightest_chan == 0:
            X, Y = Xg, Yg
            X_, Y_ = Xr, Yr
        else:
            X,Y = Xr, Yr
            X_, Y_ = Xg, Yg

        vs = self.metadata.voxelsize_nm
        x0 = vs.x * x
        y0 = vs.y * y
        
        # x0 = X.mean()
        # y0 = Y.mean()

        fitBackground = self.metadata.getOrDefault('Analysis.FitBackground', True)

        if fitBackground:
            startParameters = [A, x0, y0, 250 / 2.35, chandata.min(), .001, .001]
    
            #do the fit
            (res, cov_x, infodict, mesg, resCode) = FitModelWeighted(f_gauss2d, startParameters, chandata, chansigma, X, Y)
        else:
            startParameters = [A, x0, y0, 250 / 2.35]
    
            #do the fit
            (res, cov_x, infodict, mesg, resCode) = FitModelWeighted(f_gauss2d_no_bg, startParameters, chandata, chansigma, X, Y)

        # try to estimate errors based on the covariance matrix
        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataROI.ravel())- len(res)))
        except Exception:
            pass
        
        # normalised Chi-squared
        nchi2 = (infodict['fvec']**2).sum()/(dataROI.size - res.size)
        
        
        # evaluate the fitted model on other ROIs coordinates. We will use this as a template to calculate intensity
        # in the second channel
        fitted_model = f_gauss2d_no_bg(res[:4], X_, Y_).ravel()

        # grab the data from the dimmer channel
        chandata = dataROI[:, :, 1 - brightest_chan].ravel()
        weights = 1./(sigma[:, :, 1 - brightest_chan]**2).ravel()
        
        # the dimmer channel should be a linear combination of the brighter channel and a constant background
        A, bg = np.linalg.lstsq(np.vstack((fitted_model, np.ones_like(fitted_model))).T*weights[:,None], chandata*weights, rcond=None)[0]
        
        res = _munge_chans(res, A, bg, brightest_chan)
        if not fitErrors is None:
            fitErrors = _munge_chans(fitErrors, 0, 0, brightest_chan)

        return GaussianFitResultR(res, self.metadata, startParameters,(xslice, yslice), resCode, fitErrors, nchi2, bgs, sI)

    
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde
#from PYME.localization.FitFactories import Interpolators
#from PYME.localization.FitFactories import zEstimators

PARAMETERS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              #mde.FilenameParam('PSFFilename', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              #mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              #mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              #mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'y', choices=['x', 'y'])
              ]
              
DESCRIPTION = 'Ratiometric multi-colour 2D Gaussian fit (large shifts, imbalanced ratios).'
LONG_DESCRIPTION = '''Ratiometric multi-colour 2D Gaussian fit (large shifts). This variant of the splitter fit uses the
shiftmap to extract a different ROI in each of the colour channels, allowing chromatic shifts to be larger than for the
other splitter fits. Useful if there is a magnification difference between the two colour channels, but will perform just
as well (or better) on low shift data, as a smaller ROI can be used. Assumes background already subtracted.

This variant only fits brightest channel and is suitable for scavanged fluorescnce or other situations with very imbalanced splitting ratios'''
USE_FOR = '2D multi-colour'
