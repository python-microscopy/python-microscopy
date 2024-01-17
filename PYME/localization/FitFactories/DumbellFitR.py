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

import numpy as np
from .fitCommon import fmtSlicesUsed 
from . import FFBase 

from PYME.Analysis._fithelpers import FitModelWeighted, FitModelWeightedJac

#import pylab


##################
# Model functions
def f_dumbell(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, B, x1, y1, s, bg = p
    X = X[:,None]
    Y = Y [None,:]
    r = A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + B*np.exp(-((X-x1)**2 + (Y - y1)**2)/(2*s**2)) + bg 
    #print r.shape    
    return r

#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('B', '<f4'),
                              ('x1', '<f4'),('y1', '<f4'),
                              ('sigma', '<f4'), 
                              ('background', '<f4')]),
              ('fitError', [('A', '<f4'),
                            ('x0', '<f4'),
                            ('y0', '<f4'),
                            ('B', '<f4'),
                            ('x1', '<f4'),('y1', '<f4'),
                            ('sigma', '<f4'), 
                            ('background', '<f4')]),
              ('length', '<f4'),
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4')
              ]

def FitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0, length = 0):
    slicesUsed = fmtSlicesUsed(slicesUsed)
    #print slicesUsed

    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')
    
    res =  np.array([(metadata['tIndex'], fitResults.astype('f'), fitErr.astype('f'), length, resultCode, slicesUsed, background)], dtype=fresultdtype) 
    #print res
    return res
		

class DumbellFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_dumbell, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma)

        if False:#'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted

    def FromPoint(self, x, y, z=None, roiHalfSize=7, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background
        
        #print data.shape

        #estimate some start parameters...
        A = (data - data.min()).max() #amplitude

        vs = self.metadata.voxelsize_nm
        x0 =  vs.x*x
        y0 =  vs.y*y
        
        bgm = np.mean(background)

        startParameters = [A, x0 + 70*np.random.randn(), y0+ 70*np.random.randn(), A, x0+ 70*np.random.randn(), y0+ 70*np.random.randn(), 250/2.35, dataMean.min()]	

        
        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y)

        #try to estimate errors based on the covariance matrix
        fitErrors=None
        try:       
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception:
            pass
        
        length = np.sqrt((res[1] - res[4])**2 + (res[2] - res[5])**2)
        
        if False:
            #display for debugging purposes
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            plt.subplot(141)
            plt.imshow(dataMean)
            plt.colorbar()
            plt.subplot(142)
            plt.imshow(f_dumbell(startParameters, X, Y))
            plt.colorbar()
            plt.subplot(143)
            plt.imshow(f_dumbell(res, X, Y))
            plt.colorbar()
            plt.subplot(144)
            plt.imshow(dataMean-f_dumbell(res, X, Y))
            plt.colorbar()

        #package results
        return FitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm, length)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_dumbell(params, X, Y), X[0], Y[0], 0)


#so that fit tasks know which class to use
FitFactory = DumbellFitFactory
FitResult = FitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'Fit a "dumbell" consisting of 2 Gaussians'
LONG_DESCRIPTION = 'Fit a "dumbell" consisting of 2 Gaussians'
USE_FOR = '2D single-colour'