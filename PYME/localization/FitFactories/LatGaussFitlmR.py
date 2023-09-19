#!/usr/bin/python

##################
# LatGaussFitlmR.py
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
from . import fitCommon

from PYME.localization.lev_gfit.levmar_gfit import *


class GaussianFitResult:
    def __init__(self, fitResults, metadata, slicesUsed=None, resultCode=None, fitErr=None):
        self.fitResults = fitResults
        self.metadata = metadata
        self.slicesUsed = slicesUsed
        self.resultCode=resultCode
        self.fitErr = fitErr
    
    def A(self):
        return self.fitResults[0]

    def x0(self):
        return self.fitResults[1]

    def y0(self):
        return self.fitResults[2]

    def sigma(self):
        return self.fitResults[3]

    def background(self):
        return self.fitResults[4]

    def FWHMnm(self):
        return FWHM_CONV_FACTOR*self.fitResults[3]*self.metadata.voxelsize_nm.x

    def correctedFWHM(self, FWHM_PSF):
        return scipy.sqrt(self.FWHMnm()**2 - self.FWHM_PSF**2)

    def renderFit(self):
        vs = self.metadata.voxelsize_nm
        X = vs.x*scipy.mgrid[self.slicesUsed[0]]
        Y = vs.y*scipy.mgrid[self.slicesUsed[1]]
        return f_gauss2d(self.fitResults, X, Y)
        
def replNoneWith1(n):
    if n is None:
        return 1
    else:
        return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    if slicesUsed is None:
        slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
    else:
        slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

    if fitErr is None:
        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

    #print slicesUsed

    tIndex = metadata['tIndex']


    return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed)], dtype=fresultdtype)


class GaussianFitFactory:
    def __init__(self, data, metadata, fitfcn=None, background=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
        self.metadata = metadata
        self.background = background
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        #if type(fitfcn) == types.FunctionType: #single function provided - use numerically estimated jacobian
        #	self.solver = FitModelWeighted
        #else: #should be a tuple containing the fit function and its jacobian
        #	self.solver = FitModelWeightedJac

        
    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        #average in z
        dataMean = dataROI.mean(2)

        #generate grid to evaluate function on
        vs = self.metadata.voxelsize_nm
        vs.x*scipy.mgrid[xslice]
        vs.y*scipy.mgrid[yslice]

        #estimate some start parameters...
        A = dataMean.max() - dataMean.min() #amplitude
        
        x0 =  X.mean()
        y0 =  Y.mean()

        startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]


        #estimate errors in data
        nSlices = dataROI.shape[2]
        
        sigma = scipy.sqrt(self.metadata['CCD.ReadNoise']**2 + (self.metadata['CCD.noiseFactor']**2)*self.metadata['CCD.electronsPerCount']*dataMean/nSlices)/self.metadata['CCD.electronsPerCount']


        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, ret, cov_x, nIters, resCode) = fitGauss(startParameters, X,Y, dataMean.T.ravel(), 1.0/sigma.T.ravel())
        
        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x))
        except Exception:
            pass
        return GaussianFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        x = round(x)
        y = round(y)

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 
                    max((z - axialHalfSize), 0):min((z + axialHalfSize + 1), self.data.shape[2])]
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
