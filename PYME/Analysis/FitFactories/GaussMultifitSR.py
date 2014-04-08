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

import scipy
#from scipy.signal import interpolate
import scipy.ndimage as ndimage
#from pylab import *
import copy_reg
import numpy
#import types

from PYME.Analysis.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

def f_gauss2dSlow(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    #print X.shape
    #r = genGauss(X,Y,A,x0,y0,s,b,b_x,b_y)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    #return r
    
def f_multiGaussS(p, X, Y, s):
    #number of Gaussians to fit
    nG = len(p)/3

    r = 0.0    
    
    for i in range(nG):
        i3 = 3*i
        A, x0, y0 = p[i3:(i3+3)]
        r += A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2))
        
    return r
    
def f_multiGauss(p, X, Y, s):
    #number of Gaussians to fit
    #nG = len(p)/3

    #r = 0.0    
    
    #for i in range(nG):
    #    i3 = 3*i
    #    A, x0, y0 = p[i3:(i3+3)]
    #    r += A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2))
        
    return genMultiGauss(X, Y, p, s)
    
def f_multiGaussJ(p, X, Y, s):
    #number of Gaussians to fit
    #nG = len(p)/3

    #r = 0.0    
    
    #for i in range(nG):
    #    i3 = 3*i
    #    A, x0, y0 = p[i3:(i3+3)]
    #    r += A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2))
        
    return genMultiGaussJac(X, Y, p, s)
    
f_multiGauss.D = f_multiGaussJ

def f_gauss2d(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    #print X.shape
    r = genGauss(X,Y,A,x0,y0,s,b,b_x,b_y)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    return r

def f_gauss2dF(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y] - uses fast exponential approx"""
    A, x0, y0, s, b, b_x, b_y = p
    r = genGaussF(X,Y,A,x0,y0,s,b,b_x,b_y)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    return r

def f_j_gauss2d(p,func, d, w, X,Y):
    '''generate the jacobian for a 2d Gaussian'''
    A, x0, y0, s, b, b_x, b_y = p
    #r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
    r = genGaussJacW(X,Y,w,A,x0,y0,s,b,b_x,b_y)
    r = -r.ravel().reshape((-1,7))
    #for  i in range(7):
	#r[:, i] = r[:, i]*w
    return r.T

def f_J_gauss2d(p,X,Y):
    '''generate the jacobian for a 2d Gaussian - for use with _fithelpers.weightedJacF'''
    A, x0, y0, s, b, b_x, b_y = p
    r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
    r = r.reshape((-1, 7))
    return r

f_gauss2d.D = f_J_gauss2d


        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]), ('resultCode', '<i4'), ('nChi2', '<f4'), ('nFit', '<i4')]

def GaussianFitResultR(fitResults, metadata, resultCode=-1, fitErr=None, nChi2=0, nEvents=1):
	
	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, nChi2, nEvents)], dtype=fresultdtype) 
		

class GaussianFitFactory:
    X = None
    Y = None
    
    def __init__(self, data, metadata, fitfcn=f_multiGauss, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. '''
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if 'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted
            
        vx = 1e3*self.metadata.voxelsize.x
        vy = 1e3*self.metadata.voxelsize.y
        
        #only recalculate grid if existing one doesn't match
        if not self.X or not self.X.shape == self.data.shape[:2]:
             X,  Y = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]
             self.X = vx*X
             self.Y = vy*Y
             
#        u,v = np.mgrid[0:10., 0:10]
#        u = vx*u
#        v = vy*v
        
        gSig = float(self.metadata.getOrDefault('Analysis.PSFSigma', 130.))
        
#        self.gLUT = (vx*vy/(2*np.pi*gSig*gSig))*np.exp(-(u*u + v*v)/(2*gSig*gSig))
        self.gLUT2 = (vx*vy/(2*np.pi*gSig*gSig))*np.exp(-(vx*vx*np.arange(33.))/(2*gSig*gSig))
        self.gLUT2[-1] = 0
        
    def _gFilter(self, x, y, vals):
        ret = 0*vals
        
        x = (x/(1e3*self.metadata.voxelsize.x)).astype('i')
        y = (y/(1e3*self.metadata.voxelsize.y)).astype('i')
        
        dx = x[:,None] - x[None,:]
        #dx = np.minimum(np.abs(x[:,None] - x[None,:]), 9)
        dy = y[:,None] - y[None,:]
        #dy = np.minimum(np.abs(y[:,None] - y[None,:]), 9)
        
        
        di = dx*dx + dy*dy
        di = np.minimum(di, 32)
        
        
        
        wi = self.gLUT2[di]
        
#        import pylab
        
#        pylab.figure(figsize=(16, 5))
#        pylab.subplot(131)
#        pylab.imshow(dx)
#        pylab.subplot(132)
#        
#        pylab.imshow(di)
#        pylab.colorbar()
#        pylab.subplot(133)
#        pylab.imshow(wi)
        
#        dx = np.minimum(np.abs(x[:,None] - x[None,:]), 9)
#        dy = np.minimum(np.abs(y[:,None] - y[None,:]), 9)
#        
#        wi = self.gLUT[dx, dy]
#        
#        pylab.figure(figsize=(16, 5))
#        pylab.subplot(131)
#        pylab.imshow(dx)
#        pylab.subplot(132)
#        
#        pylab.imshow(dy)
#        pylab.colorbar()
#        pylab.subplot(133)
#        pylab.imshow(wi)
        
        ret = (vals[:,None]*wi).sum(0)
        
#        pylab.figure()
#        pylab.plot(vals)
#        pylab.plot(ret)
#        
#        ret = 0*vals
#        
#        for i in range(len(x)):
#            dx = np.minimum(np.abs(x - x[i]), 9).astype('i')
#            dy = np.minimum(np.abs(y - y[i]), 9).astype('i')
#                    
#            ret[i] =  (vals*self.gLUT[dx, dy]).sum()
#
#        pylab.plot(ret)        
#        
        return ret
        
    def _gFilter2(self, x, y, vals):        
        x = (x/(1e3*self.metadata.voxelsize.x)).astype('i')
        y = (y/(1e3*self.metadata.voxelsize.y)).astype('i')
        
            
        return NRFilter(x, y, vals, self.gLUT2)
            

    def FindAndFit(self, threshold=2, gui=False):
        #average in z
        dataMean = self.data.mean(2) - self.metadata.Camera.ADOffset

	
        #estimate errors in data
        nSlices = self.data.shape[2]
        
        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*scipy.maximum(dataMean, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount

        if not self.background == None and len(numpy.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            #average in z
            bgMean = self.background.mean(2) - self.metadata.Camera.ADOffset
            
            dataMean = dataMean - bgMean
            
        #ofind step
        import pylab
        #find pixels which are > 2 sigma above noise floor.
        #sl = ndimage.gaussian_filter(sigma, 3)
        pe = np.log(np.maximum(dataMean/sigma, .1))
        dt = dataMean > threshold*sigma
        
#        pylab.imshow(dataMean.T, interpolation='nearest')
#        pylab.colorbar()
#        pylab.figure()
#        
#        pylab.imshow(sigma.T, interpolation='nearest')
#        pylab.colorbar()
#        pylab.figure()
#        
#        pylab.imshow(pe.T, interpolation='nearest')
#        pylab.colorbar()
#        pylab.figure()
        
        
        pt = ndimage.uniform_filter(pe) > threshold
        
#        pylab.imshow(ndimage.uniform_filter(pe).T)
#        pylab.colorbar()
#        pylab.figure()
        
        dt = pt*dt
        
        #true events have correlated pixels. Look for at least 3 adjoining pixels on
        #dt = (dt*ndimage.uniform_filter(dt.astype('f'))) > 0.35
        
#        pylab.imshow(dt.T)
#        pylab.figure()
        
        dt = (dt*ndimage.uniform_filter(dt.astype('f'))) > 0.35
        
#        pylab.imshow(dt.T)
#        pylab.figure()
        
        #now hole fill and pad out around found objects
        mask = (ndimage.uniform_filter(dt.astype('f'))) > 0.1
        
#        pylab.imshow(mask.T)
#        pylab.figure()
        
        #further pad        
        #mask2 = (ndimage.uniform_filter(mask.astype('f'))) > 0.2
        
        #lab2 = 
        
        
        #pylab.figure()
#        pylab.imshow(dt.T, interpolation='nearest')
#        
#        pylab.figure()
#        pylab.imshow(mask.T, interpolation='nearest')
#        pylab.figure()
        #starting guesses
        labels, nlabels = ndimage.label(mask)
        #print nlabels, mask.sum()
        #print labels.dtype
        
        #labels = (labels*mask).astype('int32')
            
        #objSlices = ndimage.find_objects(labels)
            
        if gui:
            pylab.imshow(dataMean.T,interpolation='nearest')
            pylab.figure()
            pylab.imshow(mask.T, interpolation='nearest')
            #pylab.figure()
            
        
        if nlabels == 0:
            #the frame is empty
            resList = np.empty(0, FitResultsDType)
            return resList
            
        #nTotEvents = nlabels
        allEvents = []
        
        gSig = self.metadata.getOrDefault('Analysis.PSFSigma', 130.)
        rMax = self.metadata.getOrDefault('Analysis.ResidualMax', .25)
        
        
        
        def plotIterate(res, os, residuals, resfilt):
            pylab.figure(figsize=(20,4))
            pylab.subplot(141)
            pylab.imshow(dataMean.T,interpolation='nearest')
            pylab.contour(mask.T, [0.5], colors=['y'])
            pylab.plot(res[1::3]/70, res[2::3]/70, 'xr')
            pylab.subplot(142)
            md = self.fitfcn(res,self.X.ravel(), self.Y.ravel(), gSig).reshape(dataMean.shape)
            pylab.imshow(md.T)
            pylab.subplot(143)
            #pylab.imshow(((dataMean-md)/sigma).T, interpolation='nearest')
            rs = np.zeros_like(dataMean)
            rs[os] = residuals
            pylab.imshow(rs.T)
            #pylab.colorbar()
            pylab.subplot(144)
            rs = np.zeros_like(dataMean)
            rs[os] = resfilt
            pylab.imshow(rs.T)
            pylab.colorbar()
            
        
        #loop over objects
        for i in range(nlabels):
            #startParameters = []
            #measure position
            #x,y = ndimage.center_of_mass(im, labeledPoints, i)
            os =  labels == (i+1) #objSlices[i]           
            
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
                
                
                #residual *= (abs(X_m - res[1::3]) > 50)*(abs(Y_m - res[2::3]) > 50)
                #correlate residuals with PSF
                #resf = self._gFilter(X_m, Y_m, residual/s_m)
                resf = self._gFilter2(X_m, Y_m, residual/s_m)
                
                resmax = (resf).max()
                
                
                if gui ==2:
                   plotIterate(res, os, residual/s_m, resf)
                
                #resi = 0*dataMean
                
                #print resi[os].shape, residual.shape                
                
                #resi[os] = residual/s_m
                #resmax = ndimage.uniform_filter(resi).max()
                
                #print nchi2, resmax
        
                #prevent an infinite loop here      
        
                
                while resmax > rMax and nEvents < 10 and d_m.size > (3*(nEvents+1)):    
                    nEvents += 1
                    #print nEvents
                    
                    resI = np.argmax(residual/s_m)
                    resI = np.argmax(resf)
                    
                    startParameters = np.hstack((res,  np.array([residual[resI], X_m[resI], Y_m[resI]])))
                    
                    (res_n, cov_x_n, infodict_n, mesg_n, resCode_n) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
                    
                    #test for convergence - no convergence = abandon and unwind back to previous
                    if cov_x_n == None:
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
                        print nEvents, nchi2, resmax, resCode#, cov_x
                    
                    #resi = 0*dataMean
                    #resi[os][:] = residual/s_m
                    #resmax = ndimage.uniform_filter(resi).max()
                    
                    
                
            
    
                #work out the errors
                fitErrors=None
                try:       
                    fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(d_m)- len(res)))
                except Exception, e:
                    pass
                #print res, fitErrors, resCode
                #recreate a list of events in the desired format
                resList = np.empty(nEvents, FitResultsDType)
                for i in range(nEvents):
                    i3 = 3*i
                    i31 = i3 + 3
                    
                    if not fitErrors == None:            
                        resList[i] = GaussianFitResultR(res[i3:i31], self.metadata, resCode, fitErrors[i3:i31], nchi2, nEvents)
                    else:
                        resList[i] = GaussianFitResultR(res[i3:i31], self.metadata, resCode, None, nchi2, nEvents)
                        
                allEvents.append(resList)
        
        return np.hstack(allEvents)
        
    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        X = 1e3*md.voxelsize.x*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = 1e3*md.voxelsize.y*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_gauss2d(params, X, Y), X[0], Y[0], 0)


   
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

#this means that factory is reponsible for it's own object finding and implements
#a GetAllResults method that returns a list of localisations
MULTIFIT=True

import PYME.Analysis.MetaDataEdit as mde
#from PYME.Analysis.FitFactories import Interpolators
#from PYME.Analysis.FitFactories import zEstimators

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