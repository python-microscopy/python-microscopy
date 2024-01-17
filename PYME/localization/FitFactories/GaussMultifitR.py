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

import scipy.ndimage as ndimage
import numpy as np

from PYME.localization.cModels.gauss_app import *
from PYME.Analysis._fithelpers import *
from .fitCommon import fmtSlicesUsed



def f_gauss2dSlow(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    return A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y

    
def f_multiGaussS(p, X, Y, s):
    #number of Gaussians to fit
    nG = len(p)/3

    r = 0.0    
    
    for i in range(nG):
        i3 = 3*i
        A, x0, y0 = p[i3:(i3+3)]
        r += A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2))
        
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




fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]),
              ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4')]), 
              ('resultCode', '<i4')]

def GaussianFitResultR(fitResults, metadata, resultCode=-1, fitErr=None):	
    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')

    tIndex = metadata['tIndex']

    return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode)], dtype=fresultdtype)


class GaussianFitFactory:
    X = None
    Y = None
    
    def __init__(self, data, metadata, fitfcn=f_multiGauss, background=None, **kwargs):
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
             X,  Y = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]
             vs = self.metadata.voxelsize_nm
             self.X = vs.x*X
             self.Y = vs.y*Y
            

    def FindAndFit(self, threshold=2):
        #average in z
        dataMean = self.data.mean(2)


        #estimate errors in data
        nSlices = self.data.shape[2]
        
        sigma = np.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*np.maximum(dataMean, 1)/nSlices)/self.metadata['Camera.ElectronsPerCount']

        if not self.background is None and len(np.shape(self.background)) > 1 and self.metadata.getOrDefault('Analysis.subtractBackground', True):
            #average in z
            bgMean = self.background.mean(2)
            
            dataMean = dataMean - bgMean
            
        #ofind step
        # import pylab
        #find pixels which are > 2 sigma above noise floor.
        dt = dataMean > threshold*sigma
        
#        pylab.imshow(dt.T)
#        pylab.figure()
        
        #true events have correlated pixels. Look for at least 3 adjoining pixels on
        dt = (dt*ndimage.uniform_filter(dt.astype('f'))) > 0.35
        
        dt = (dt*ndimage.uniform_filter(dt.astype('f'))) > 0.35
        
        #now hole fill and pad out around found objects
        mask = (ndimage.uniform_filter(dt.astype('f'))) > 0.1
        
        
        #pylab.figure()
#        pylab.imshow(dt.T, interpolation='nearest')
#        
#        pylab.figure()
#        pylab.imshow(mask.T, interpolation='nearest')
#        pylab.figure()
        #starting guesses
        labels, nlabels = ndimage.label(mask)
        print((nlabels, mask.sum()))
            
        objSlices = ndimage.find_objects(labels)
        
        startParameters = []
        
        #loop over objects
        for i in range(nlabels):
            #measure position
            #x,y = ndimage.center_of_mass(im, labeledPoints, i)
            imO = dataMean[objSlices[i]]
            imOs = imO.sum()
            x = (self.X[objSlices[i]]*imO).sum()/imOs
            y = (self.Y[objSlices[i]]*imO).sum()/imOs
            
            A = imO.max()

            #and add to list
            startParameters += [A, x, y]
            
        nEvents = nlabels

        #startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]
        
        if nlabels == 0:
            #the frame is empty
            resList = np.empty(nEvents, FitResultsDType)
            return resList
            
        d_m = dataMean[mask]
        s_m = sigma[mask]
        X_m = self.X[mask]
        Y_m = self.Y[mask]
        
        gSig = self.metadata.getOrDefault('Analysis.PSFSigma', 105.)

        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        #print self.X[mask]
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
        
#        ft = 0*dataMean
#        ft[mask] = self.fitfcn(res, self.X[mask], self.Y[mask], self.metadata['Analysis.PSFSigma'])
#        pylab.imshow(ft, interpolation='nearest')
#        pylab.colorbar()
#        pylab.figure()
        
        #return []
        
        residual = d_m - self.fitfcn(res, X_m, Y_m, gSig)
        
        nchi2 = ((residual/s_m)**2).mean()
        resmax = (residual/s_m).max()
        
        print((nchi2, resmax))

        refinementCount = 0  #prevent an infinite loop here      
        
        while resmax > 5 and refinementCount < 10:    
            nEvents += 1
            refinementCount += 1
            
            resI = np.argmax(residual/s_m)
            
            startParameters = np.hstack((res,  np.array([residual[resI], X_m[resI], Y_m[resI]])))
            
            (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, d_m, s_m, X_m, Y_m, gSig)
                
            residual = d_m - self.fitfcn(res, X_m, Y_m, gSig)
            
            nchi2 = ((residual/s_m)**2).mean()
            resmax = (residual/s_m).max()
            
            print((nchi2, resmax))
            
            

        #pylab.imshow(dataMean, interpolation='nearest')
        #pylab.colorbar()
        #pylab.figure()        
        
#        resi = 0*dataMean
#        resi[mask] = residual
#        pylab.imshow(resi, interpolation='nearest')
#        pylab.colorbar()
#        pylab.figure()
        

        #work out the errors
        fitErrors=None
        try:       
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception as e:
            pass
        #print res, fitErrors, resCode
        #recreate a list of events in the desired format
        resList = np.empty(nEvents, FitResultsDType)
        for i in range(nEvents):
            i3 = 3*i
            i31 = i3 + 3
            
            if not fitErrors is None:
                resList[i] = GaussianFitResultR(res[i3:i31], self.metadata, resCode, fitErrors[i3:i31])
            else:
                resList[i] = GaussianFitResultR(res[i3:i31], self.metadata, resCode, None)
        
        return resList
        
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
