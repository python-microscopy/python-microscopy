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
#import copy_reg
import numpy
#import types

from PYME.localization.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *

# def pickleSlice(slice):
#         return unpickleSlice, (slice.start, slice.stop, slice.step)
#
# def unpickleSlice(start, stop, step):
#         return slice(start, stop, step)
#
# copy_reg.pickle(slice, pickleSlice, unpickleSlice)

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
    """generate the jacobian for a 2d Gaussian"""
    A, x0, y0, s, b, b_x, b_y = p
    #r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
    r = genGaussJacW(X,Y,w,A,x0,y0,s,b,b_x,b_y)
    r = -r.ravel().reshape((-1,7))
    #for  i in range(7):
    #r[:, i] = r[:, i]*w
    return r.T

def f_J_gauss2d(p,X,Y):
    """generate the jacobian for a 2d Gaussian - for use with _fithelpers.weightedJacF"""
    A, x0, y0, s, b, b_x, b_y = p
    r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
    r = r.reshape((-1, 7))
    return r

f_gauss2d.D = f_J_gauss2d


        
def replNoneWith1(n):
    if n is None:
        return 1
    else:
        return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]), ('resultCode', '<i4'), ('nChi2', '<f4'), ('nFit', '<i4')]

def GaussianFitResultR(fitResults, metadata, resultCode=-1, fitErr=None, nChi2=0, nEvents=1):

    if fitErr is None:
        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

    #print slicesUsed

    tIndex = metadata['tIndex']


    return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, nChi2, nEvents)], dtype=fresultdtype)


class GaussianFitFactory:
    X = None
    Y = None
    
    def __init__(self, data, metadata, fitfcn=f_multiGauss, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if 'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted
        
        #only recalculate grid if existing one doesn't match
        if not self.X or not self.X.shape == self.data.shape[:2]:
             X,  Y, C = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1], 0:2]
             vs = self.metadata.voxelsize_nm
             self.X = vs.x*X
             self.Y = vs.y*Y
             self.C = C.astype('int')
            

    def FindAndFit(self, threshold=2, gui=False):
        #average in z
        data = self.data


        #estimate errors in data
        nSlices = self.data.shape[2]
        
        sigma = scipy.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*scipy.maximum(data, 1)/nSlices)/self.metadata['Camera.ElectronsPerCount']

        if not self.background is None and len(numpy.shape(self.background)) > 1 and self.metadata.getOrDefault('Analysis.subtractBackground', True):
            #average in z
            bgM = self.background
            
            data = data - bgM
            
        #ofind step
        #import pylab
        #find pixels which are > 2 sigma above noise floor.
        #sl = ndimage.gaussian_filter(sigma, 3)
        pe = np.log(np.maximum(data/sigma, .1))
        dt = data > threshold*sigma
        
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
        
        #collapse to 
        
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
            import matplotlib.pyplot as plt
            plt.imshow(data.T,interpolation='nearest')
            plt.figure()
            plt.imshow(mask.T, interpolation='nearest')

        if nlabels == 0:
            #the frame is empty
            resList = np.empty(0, FitResultsDType)
            return resList
            
        #nTotEvents = nlabels
        allEvents = []
        
        gSig = self.metadata.getOrDefault('Analysis.PSFSigma', 105.)
        
        #loop over objects
        for i in range(nlabels):
            #startParameters = []
            #measure position
            #x,y = ndimage.center_of_mass(im, labeledPoints, i)
            os =  labels == (i+1) #objSlices[i]           
            
            imO = data[os]
            if imO.size > 5:
                imOs = imO.sum()
                d_m = data[os].ravel()
                s_m = sigma[os].ravel()
                C_m = self.C[os].ravel()
                X_m = self.X[os].ravel()
                Y_m = self.Y[os].ravel()
                
                x = (X_m*d_m).sum()/imOs
                y = (Y_m*d_m).sum()/imOs
                
                #correct for chromatic shift
                DeltaX = self.metadata['chroma.dx'].ev(x_, y_)
                DeltaY = self.metadata['chroma.dy'].ev(x_, y_)
                
                A = imO.max()
    
                #and add to list
                startParameters = [A, x, y]
                
                nEvents = 1
                    
                
        
                #do the fit
                
                (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
                
                residual = d_m - self.fitfcn(res, X_m, Y_m, gSig)
                
                nchi2 = ((residual/s_m)**2).mean()
                resmax = (residual/s_m).max()
                
                #resi = 0*dataMean
                
                #print resi[os].shape, residual.shape                
                
                #resi[os] = residual/s_m
                #resmax = ndimage.uniform_filter(resi).max()
                
                #print nchi2, resmax
        
                 #prevent an infinite loop here      
                
                while resmax > 2 and nEvents < 10 and d_m.size > (3*(nEvents+1)):    
                    nEvents += 1
                    #print nEvents
                    
                    resI = np.argmax(residual/s_m)
                    
                    startParameters = np.hstack((res,  np.array([residual[resI], X_m[resI], Y_m[resI]])))
                    
                    (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
                        
                    residual = d_m - self.fitfcn(res, X_m, Y_m, gSig)
                    
                    nchi2 = ((residual/s_m)**2).mean()
                    resmax = (residual/s_m).max()
                    
                    #resi = 0*dataMean
                    #resi[os][:] = residual/s_m
                    #resmax = ndimage.uniform_filter(resi).max()
                    
                    #print nchi2, resmax
                
            
    
                #work out the errors
                fitErrors=None
                try:       
                    fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(d_m)- len(res)))
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
        X = vs.x*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

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
              mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              #mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              #mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              #mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'y', choices=['x', 'y'])
              ]
