#!/usr/bin/python

##################
# LatObjFindFR.py
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
import copy_reg
import numpy
import types

from PYME.Analysis.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)

#def f_gauss2d(p, X, Y):
#    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
#    A, x0, y0, s, b, b_x, b_y = p
#    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
#    r = genGauss(X,Y,A,x0,y0,s,b,b_x,b_y)
#    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
#    return r
#
#def f_gauss2dF(p, X, Y):
#    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y] - uses fast exponential approx"""
#    A, x0, y0, s, b, b_x, b_y = p
#    r = genGaussF(X,Y,A,x0,y0,s,b,b_x,b_y)
#    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
#    return r
#
#def f_j_gauss2d(p,func, d, w, X,Y):
#    '''generate the jacobian for a 2d Gaussian'''
#    A, x0, y0, s, b, b_x, b_y = p
#    #r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
#    r = genGaussJacW(X,Y,w,A,x0,y0,s,b,b_x,b_y)
#    r = -r.ravel().reshape((-1,7))
#    #for  i in range(7):
#	#r[:, i] = r[:, i]*w
#    return r.T
#
#def f_J_gauss2d(p,X,Y):
#    '''generate the jacobian for a 2d Gaussian - for use with _fithelpers.weightedJacF'''
#    A, x0, y0, s, b, b_x, b_y = p
#    r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
#    r = r.reshape((-1, 7))
#    return r.T
#
#f_gauss2d.D = f_J_gauss2d


#class GaussianFitResult:
#    def __init__(self, fitResults, metadata, slicesUsed=None, resultCode=None, fitErr=None):
#        self.fitResults = fitResults
#        self.metadata = metadata
#        self.slicesUsed = slicesUsed
#        self.resultCode=resultCode
#        self.fitErr = fitErr
#
#    def A(self):
#        return self.fitResults[0]
#
#    def x0(self):
#        return self.fitResults[1]
#
#    def y0(self):
#        return self.fitResults[2]
#
#    def sigma(self):
#        return self.fitResults[3]
#
#    def background(self):
#        return self.fitResults[4]
#
#    def FWHMnm(self):
#        return FWHM_CONV_FACTOR*self.fitResults[3]*self.metadata.voxelsize.x*1e3
#
#    def correctedFWHM(self, FWHM_PSF):
#        return scipy.sqrt(self.FWHMnm()**2 - self.FWHM_PSF**2)
#
#    def renderFit(self):
#	X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[self.slicesUsed[0]]
#        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[self.slicesUsed[1]]
#        return f_gauss2d(self.fitResults, X, Y)
        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else: 		
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed)], dtype=fresultdtype) 
		

class GaussianFitFactory:
    def __init__(self, data, metadata, fitfcn=None, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in 
        metadata. '''
        self.data = data
        self.metadata = metadata
	self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
	if type(fitfcn) == types.FunctionType: #single function provided - use numerically estimated jacobian
		self.solver = FitModelWeighted
	else: #should be a tuple containing the fit function and its jacobian
		self.solver = FitModelWeightedJac
		
        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        if (z == None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()
	
        x_ = round(x)
        y_ = round(y)
	
        xslice = slice(max((x_ - roiHalfSize), 0),min((x_ + roiHalfSize + 1),self.data.shape[0]))
        yslice = slice(max((y_ - roiHalfSize), 0),min((y_ + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(max((z - axialHalfSize), 0),min((z + axialHalfSize + 1), self.data.shape[2]))

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        #average in z
        dataMean = dataROI.mean(2) - self.metadata.Camera.ADOffset

        #estimate some start parameters...
        A = dataMean.max() - dataMean.min() #amplitude
        
        x0 =  x*1e3*self.metadata.voxelsize.x
        y0 =  y*1e3*self.metadata.voxelsize.y

        startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]
        
        fitErrors=None
        
        return GaussianFitResultR(scipy.array(startParameters), self.metadata, (xslice, yslice, zslice), 0, fitErrors)
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
