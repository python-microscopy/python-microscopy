#!/usr/bin/python

##################
# LatGaussFitFR.py
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
import scipy.ndimage as ndimage
import numpy as np
from .fitCommon import fmtSlicesUsed

from PYME.localization.cModels.gauss_app import *
from PYME.Analysis._fithelpers import *

    
def f_multiGaussS(p, X, Y, s):
    #number of Gaussians to fit
    nG = len(p)/3
    
    nm = 1./(s*s*2*np.pi)

    r = 0.0    
    
    for i in range(nG):
        i3 = 3*i
        A, x0, y0 = p[i3:(i3+3)]
        r += A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2))*nm
        
    return r
    
def f_multiGauss(p, X, Y, s):    
    return genMultiGauss(X, Y, p, s)
    
def f_multiGaussJ(p, X, Y, s):
    return genMultiGaussJac(X, Y, p, s)
    
f_multiGauss.D = f_multiGaussJ

        



fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]),
              ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]), 
              ('resultCode', '<i4'), 
              ('nChi2', '<f4'), 
              ('nFit', '<i4')]

def GaussianFitResultR(fitResults, metadata, resultCode=-1, fitErr=None, nChi2=0, nEvents=1):	
    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')
        
    tIndex = metadata.getOrDefault('tIndex', 0)
    
    return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, nChi2, nEvents)], dtype=fresultdtype) 
		

class GaussianFitFactory:
    X = None
    Y = None
    
    def __init__(self, data, metadata, fitfcn=f_multiGauss, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
        self.background = background
        self.noiseSigma = noiseSigma
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if 'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted

        vx, vy, _ = self.metadata.voxelsize_nm
        
        #only recalculate grid if existing one doesn't match
        if not self.X or not self.X.shape == self.data.shape[:2]:
             X,  Y = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]
             self.X = vx*X
             self.Y = vy*Y
             
        gSig = float(self.metadata.getOrDefault('Analysis.PSFSigma', 130.))
        
        self.gLUT2 = (vx*vy/(2*np.pi*gSig*gSig))*np.exp(-(vx*vx*np.arange(33.))/(2*gSig*gSig))
        self.gLUT2[-1] = 0
        
#    def _gFilter(self, x, y, vals):
#        ret = 0*vals
#        
#        x = (x/(1e3*self.metadata.voxelsize.x)).astype('i')
#        y = (y/(1e3*self.metadata.voxelsize.y)).astype('i')
#        
#        dx = x[:,None] - x[None,:]
#        
#        dy = y[:,None] - y[None,:]
#        
#        di = dx*dx + dy*dy
#        di = np.minimum(di, 32)
#        
#        wi = self.gLUT2[di]
#        
#        ret = (vals[:,None]*wi).sum(0)
#        
#        return ret
        
    def _gFilter2(self, x, y, vals):
        vs = self.metadata.voxelsize_nm
        x = (x/(vs.x)).astype('i')
        y = (y/(vs.y)).astype('i')
        
            
        return NRFilter(x, y, vals, self.gLUT2)
            

    def FindAndFit(self, threshold=2, gui=False):
        #average in z
        dataMean = self.data.mean(2)

	
        #estimate errors in data
        nSlices = self.data.shape[2]
        
        if self.noiseSigma is None:        
            sigma = np.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*np.maximum(dataMean, 1)/nSlices)/self.metadata['Camera.ElectronsPerCount']
        else:
            sigma = self.noiseSigma

        #estimate and subtract background by assuming that the background pattern is 
        #spatially separable into components along each of the two image axes.
        #this should work for background caused by an illumination pattern (e.g. lines)
        #which fullfills this assumption.
        dm_mean1 = np.mean(dataMean, 1)
        bgEst = dataMean.mean(0)[None,:]*(dm_mean1/dm_mean1.mean())[:, None]
        
        dataMean = dataMean - bgEst
            
        #ofind step
        # import pylab
        import matplotlib.pyplot as plt
        #find pixels which are above noise floor.
        pe = np.log(np.maximum(dataMean/sigma, .1))
        dt = dataMean > threshold*sigma
                
        pt = ndimage.uniform_filter(pe) > threshold
                
        dt = pt*dt
        
        #true events have correlated pixels. Look for at least 3 adjoining pixels on        
        dt = (dt*ndimage.uniform_filter(dt.astype('f'))) > 0.35
                
        #now hole fill and pad out around found objects
        mask = (ndimage.uniform_filter(dt.astype('f'))) > 0.1
        
        #label contiguous regions of pixels
        labels, nlabels = ndimage.label(mask)
            
        if gui:
            plt.imshow(dataMean.T,interpolation='nearest')
            plt.figure()
            plt.imshow(mask.T, interpolation='nearest')
            
        
        if nlabels == 0:
            #the frame is empty
            resList = np.empty(0, FitResultsDType)
            return resList
            
        allEvents = []
        
        gSig = self.metadata.getOrDefault('Analysis.PSFSigma', 130.)
        rMax = self.metadata.getOrDefault('Analysis.ResidualMax', .25)
        
        
        
        def plotIterate(res, os, residuals, resfilt):
            plt.figure(figsize=(20,4))
            plt.subplot(141)
            plt.imshow(dataMean.T,interpolation='nearest')
            plt.contour(mask.T, [0.5], colors=['y'])
            plt.plot(res[1::3]/70, res[2::3]/70, 'xr')
            plt.subplot(142)
            md = self.fitfcn(res,self.X.ravel(), self.Y.ravel(), gSig).reshape(dataMean.shape)
            plt.imshow(md.T)
            plt.subplot(143)
            #plt.imshow(((dataMean-md)/sigma).T, interpolation='nearest')
            rs = np.zeros_like(dataMean)
            rs[os] = residuals
            plt.imshow(rs.T)
            #plt.colorbar()
            plt.subplot(144)
            rs = np.zeros_like(dataMean)
            rs[os] = resfilt
            plt.imshow(rs.T)
            plt.colorbar()
            
        
        #loop over objects
        for i in range(nlabels):
            os =  labels == (i+1)            
            
            imO = dataMean[os]
            if imO.size > 5:
                imOs = imO.sum()
                x = (self.X[os]*imO).sum()/imOs
                y = (self.Y[os]*imO).sum()/imOs
                
                A = imO.max()
    
                #and add to list
                startParameters = [A, x, y]
                
                nEvents = 1
                    
                d_m = dataMean[os].ravel()
                s_m = sigma[os].ravel()
                X_m = self.X[os].ravel()
                Y_m = self.Y[os].ravel()
        
                #do the fit
                
                (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
                
                residual = d_m - self.fitfcn(res, X_m, Y_m, gSig)
                
                nchi2 = ((residual/s_m)**2).mean()
                
                #correlate residuals with PSF
                #resf = self._gFilter(X_m, Y_m, residual/s_m)
                resf = self._gFilter2(X_m, Y_m, residual/s_m)
                
                resmax = (resf).max()
                
                
                if gui ==2:
                   plotIterate(res, os, residual/s_m, resf)
                
                
        
                #Continue adding events while the residuals are too high and the fit well defined
                while resmax > rMax and nEvents < 10 and d_m.size > (3*(nEvents+1)):    
                    nEvents += 1
                    #print nEvents
                    
                    resI = np.argmax(residual/s_m)
                    resI = np.argmax(resf)
                    
                    startParameters = np.hstack((res,  np.array([residual[resI], X_m[resI], Y_m[resI]])))
                    
                    (res_n, cov_x_n, infodict_n, mesg_n, resCode_n) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
                    
                    #test for convergence - no convergence = abandon and unwind back to previous
                    if cov_x_n is None:
                        nEvents -= 1
                        break
                    else: # fit converged - continue
                        res, cov_x, infodict, mesg, resCode = (res_n, cov_x_n, infodict_n, mesg_n, resCode_n)
                        
                    residual = d_m - self.fitfcn(res, X_m, Y_m, gSig)
                    
                    nchi2 = ((residual/s_m)**2).mean()
                    #resmax = (residual/s_m).max()
                    
                    resf = self._gFilter2(X_m, Y_m, residual/s_m)
                    resmax = (resf).max()
                    
                    if gui ==2:
                        plotIterate(res, os, residual/s_m, resf)
                        print((nEvents, nchi2, resmax, resCode))#, cov_x
                    
                    
    
                #work out the errors
                fitErrors=None
                try:       
                    fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(d_m)- len(res)))
                except Exception as e:
                    pass
                #print res, fitErrors, resCode
                #recreate a list of events in the desired format
                resList = np.empty(nEvents, FitResultsDType)
                for j in range(nEvents):
                    i3 = 3*j
                    i31 = i3 + 3
                    
                    if not fitErrors is None:            
                        resList[i] = GaussianFitResultR(res[i3:i31], self.metadata, resCode, fitErrors[i3:i31], nchi2, nEvents)
                    else:
                        resList[i] = GaussianFitResultR(res[i3:i31], self.metadata, resCode, None, nchi2, nEvents)
                        
                allEvents.append(resList)
        
        return np.hstack(allEvents)
        
    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_gauss2d(params, X, Y), X[0], Y[0], 0)


   
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

#this means that factory is reponsible for it's own object finding and implements
#a GetAllResults method that returns a list of localisations
MULTIFIT=True

import PYME.localization.MetaDataEdit as mde
#from PYME.localization.FitFactories import Interpolators
#from PYME.localization.FitFactories import zEstimators

PARAMETERS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              #mde.FilenameParam('PSFFilename', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              #mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              mde.FloatParam('Analysis.PSFSigma', u'PSF \u03c3  [nm]:', 130.),
              mde.FloatParam('Analysis.ResidualMax', 'Max residual:', 0.25),
              #mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              #mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'y', choices=['x', 'y'])
              ]
              
DESCRIPTION = '2D Gaussian multi-emitter fitting'
LONG_DESCRIPTION = '2D Gaussian multi-emitter fitting: Fits x, y and I, assuming background is already subtracted. Uses it\'s own object detection routine'   
USE_FOR = '2D multi-emitter'
