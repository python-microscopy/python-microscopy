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
#import scipy.ndimage as ndimage
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
        return FWHM_CONV_FACTOR*self.fitResults[3]*self.metadata.voxelsize.x*1e3

    def correctedFWHM(self, FWHM_PSF):
        return scipy.sqrt(self.FWHMnm()**2 - self.FWHM_PSF**2)

    def renderFit(self):
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[self.slicesUsed[0]]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[self.slicesUsed[1]]
        return f_gauss2d(self.fitResults, X, Y)
        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]),
              ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4')
              ]

########################################
## start map reading and processing code
########################################

# map reading and processing code
# this should really go in its own module
# currently this typically fails on all but the acquiring machine (because other machines do not have the maps)
import warnings
import os
from PYME.gohlke import tifffile as tif

def readmaps():
        mapdir = os.getenv('PYMEZYLAMAPDIR',
                           default='C:/python-microscopy-exeter/PYME/Analysis/FitFactories/')
        maps = {}
        try:
                maps['offset'] = tif.imread(os.path.join(mapdir,'offset.tif'))
                maps['variance'] = tif.imread(os.path.join(mapdir,'variance.tif'))
                gain = tif.imread(os.path.join(mapdir,'gain.tif'))
                maps['gain'] =  gain / gain.mean()
                print 'loader V2: loaded Zyla maps'
        except:
                warnings.warn('cannot load Zyla property maps')
                pass
        return maps

def mapROI(map,md):
        return map[md.Camera.ROIPosX-1:md.Camera.ROIPosX-1+md.Camera.ROIWidth,
                   md.Camera.ROIPosY-1:md.Camera.ROIPosY-1+md.Camera.ROIHeight]        

# this is the general idea
# should probably go into a separate module to manipulate and generate maps
def readnoiseBlemish(rn,offs,gain,rnMax=10.3,offsMaxdev=20,gainMaxdev=0.3, blemishVal=1e7):
        offmedian = numpy.median(offs)
        rn = rn.copy()
        rn[rn > rnMax] = blemishVal
        rn[offs>offmedian+offsMaxdev] = blemishVal
        rn[gain > 1.0+gainMaxdev] = blemishVal
        rn[gain < 1.0-gainMaxdev] = blemishVal
        return rn

######################################
## end map reading and processing code
######################################

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else: 		
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed, background)], dtype=fresultdtype) 
		

class GaussianFitFactory:
    maps = None  # maps are stored in a class variable and only read once when first object is instantiated
    def __init__(self, data, metadata, fitfcn=f_gauss2d, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. '''

        #self.Zyla_offset = numpy.loadtxt('C:/python-microscopy-exeter/PYME/Analysis/FitFactories/offset.txt')
        #self.Zyla_variance = numpy.loadtxt('C:/python-microscopy-exeter/PYME/Analysis/FitFactories/variance.txt')
        #self.Zyla_gain = numpy.loadtxt('C:/python-microscopy-exeter/PYME/Analysis/FitFactories/gain.txt')

        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if False:#'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted

        # read camera maps and make sub regions matching ROI of this series
        if self.__class__.maps is None: # read only once for this class
                self.__class__.maps = readmaps()
        maps = self.__class__.maps
        self.region_offset = mapROI(maps['offset'],self.metadata)
        region_variance = mapROI(maps['variance'],self.metadata)
        # note: readnoise must be in units of e- ; this should really have taken place by storing readnoise directly rather than variance
        self.region_readnoise = numpy.sqrt(region_variance)*self.metadata.Camera.ElectronsPerCount
        self.region_gain = mapROI(maps['gain'],self.metadata)
        self.region_readnoiseB = readnoiseBlemish(self.region_readnoise,self.region_offset,self.region_gain)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        if (z == None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        x0 = x
        y0 = y
        x = round(x)
        y = round(y)

        #return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]),
        #            max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]),
        #            max((z - axialHalfSize), 0):min((z + axialHalfSize + 1), self.data.shape[2])]

        xslice = slice(max((x - roiHalfSize), 0),min((x + roiHalfSize + 1),self.data.shape[0]))
        yslice = slice(max((y - roiHalfSize), 0),min((y + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(max((z - axialHalfSize), 0),min((z + axialHalfSize + 1), self.data.shape[2]))
        
    #def __getitem__(self, key):
        #print key
        #xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        offsetROI = self.region_offset[xslice, yslice]
        readnoiseROI = self.region_readnoiseB[xslice, yslice] # we use the readnoise with blemish pixels which are effectively 'missing data'
        gainROI = self.region_gain[xslice, yslice]

        # average in z
        # dataMean = dataROI.mean(2) - self.metadata.Camera.ADOffset
	# meangain = 1/0.28 # this should really be set elsewhere and also needs a less confusing name

        dataMean = (dataROI.mean(2) - offsetROI)/gainROI # gainROI = gain variation; gain variation = raw (raw data before filter)/ rawf (gaussian filtered data)

        #print (dataMean.shape == region_offset.shape, dataMean.shape == region_readnoise.shape, dataMean.shape == region_gain.shape)
        #print (self.data.shape, region_offset.shape, region_readnoise.shape, region_gain.shape)

        #generate grid to evaluate function on        
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[xslice]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]

        #estimate some start parameters...
        A = dataMean.max() - dataMean.min() #amplitude

        x0 =  1e3*self.metadata.voxelsize.x*x0
        y0 =  1e3*self.metadata.voxelsize.y*y0
        
        #x0 =  X.mean()
        #y0 =  Y.mean()

	
        #estimate errors in data
        nSlices = dataROI.shape[2]
        
        #sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*scipy.maximum(dataMean, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount
        sigma = scipy.sqrt(readnoiseROI**2 + scipy.maximum(dataMean, 1)*
                           self.metadata.Camera.ElectronsPerCount/nSlices)/self.metadata.Camera.ElectronsPerCount


        bgm = 0

        if not self.background == None and len(numpy.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, zslice]

            #average in z
            bgMean = bgROI.mean(2) - self.metadata.Camera.ADOffset
            
            bgm = bgMean.mean()
            
            dataMean = dataMean - bgMean

        startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]
	

        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y)

        
        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception, e:
            pass
        return GaussianFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm)

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
