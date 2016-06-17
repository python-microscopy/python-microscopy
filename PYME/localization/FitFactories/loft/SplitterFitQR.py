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
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types

#import PYME.Analysis.points.twoColour as twoColour

from PYME.localization.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)


def f_gauss2d2c(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR, b_x, b_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    #force amilitude to be positive
    Ag = sqrt(Ag**2 + 1) - 1
    Ar = sqrt(Ar**2 + 1) - 1   
    
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....
    
    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)

def f_gauss2d2ccb(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    
    #force amilitude to be positive
    Ag = sqrt(Ag**2 + 1) - 1
    Ar = sqrt(Ar**2 + 1) - 1   
    
    r = genGauss(Xr,Yr,Ar,x0,y0,s,0,0,0)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,0,0,0)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....

    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)


        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4')]),
              ('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4')]),
              ('startParams', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4')]), 
              ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def GaussianFitResultR(fitResults, metadata, startParams, slicesUsed=None, resultCode=-1, fitErr=None):
	if slicesUsed == None:
		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
	else: 		
		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

	if fitErr == None:
		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex

	#print fitResults.dtype
	#print fitErr.dtype
	#print fitResults
	#print fitErr
	#print tIndex
	#print slicesUsed
	#print resultCode


	return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), resultCode, slicesUsed)], dtype=fresultdtype) 
		
def splWrap(*args):
    #print ''
    #args = args[:]
    return splitGaussWeightedMisfit(*args).copy()

class GaussianFitFactory:
    def __init__(self, data, metadata, fitfcn=f_gauss2d2ccb, background=None):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)

        if False:#'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else:
            self.solver = FitModelWeighted
            
    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on        
        Xg = 1e3*md.voxelsize.x*scipy.mgrid[slice(-roiHalfSize,roiHalfSize + 1)]
        Yg = 1e3*md.voxelsize.y*scipy.mgrid[slice(-roiHalfSize,roiHalfSize + 1)]

        #generate a corrected grid for the red channel      
        DeltaX = md.chroma.dx.ev(x, y)
        DeltaY = md.chroma.dy.ev(x, y)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY
        
        #print DeltaX, DeltaY


        return f_gauss2d2ccb(params, Xg, Yg, Xr, Yr), Xg.ravel()[0], Yg.ravel()[0], 0
		
        
    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

        #average in z
        #dataMean = dataROI.mean(2) - self.metadata.CCD.ADOffset

        #generate grid to evaluate function on        
        Xg = 1e3*self.metadata.voxelsize.x*scipy.mgrid[xslice]
        Yg = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata.chroma.dx,self.metadata.chroma.dy)
        x_ = Xg.mean() + (self.metadata.Camera.ROIPosX - 1)*1e3*self.metadata.voxelsize.x
        y_ = Yg.mean() + (self.metadata.Camera.ROIPosY - 1)*1e3*self.metadata.voxelsize.y
        DeltaX = self.metadata.chroma.dx.ev(x_, y_)
        DeltaY = self.metadata.chroma.dy.ev(x_, y_)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY

        #print DeltaX, DeltaY
        #print DeltaY

        

        #startParameters = [Ag, Ar, x0, y0, 250/2.35, dataROI[:,:,0].min(),dataROI[:,:,1].min(), .001, .001]

	
        #estimate errors in data
        nSlices = 1#dataROI.shape[2]
        
        #sigma = scipy.sqrt(self.metadata.CCD.ReadNoise**2 + (self.metadata.CCD.noiseFactor**2)*self.metadata.CCD.electronsPerCount*self.metadata.CCD.EMGain*dataROI)/self.metadata.CCD.electronsPerCount
        sigma = scipy.sqrt(self.metadata.Camera.ReadNoise**2 + (self.metadata.Camera.NoiseFactor**2)*self.metadata.Camera.ElectronsPerCount*self.metadata.Camera.TrueEMGain*scipy.maximum(dataROI, 1)/nSlices)/self.metadata.Camera.ElectronsPerCount


        if not self.background == None and len(numpy.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

            dataROI = dataROI - bgROI
            
        #estimate some start parameters...
        Ag = dataROI[:,:,0].max() #- dataROI[:,:,0].min() #amplitude
        Ar = dataROI[:,:,1].max() #- dataROI[:,:,1].min() #amplitude

        #figure()
        #imshow(dataROI[:,:,1], interpolation='nearest')

        #print Ag
        #print Ar

        x0 =  Xg.mean()
        y0 =  Yg.mean()

        startParameters = numpy.array([Ag, Ar, x0, y0, 250/2.35])
	
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr, buf)
        buf = numpy.zeros(dataROI.size)
        #(res, cov_x, infodict, mesg, resCode) = FitWeightedMisfitFcn(splitGaussWeightedMisfit, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr)
        (res, cov_x, infodict, mesg, resCode) = FitWeightedMisfitFcn(splWrap, startParameters, dataROI, sigma, Xg, Yg, Xr, Yr, buf)

        #print res        
        #we map Ag and Ar to ensure positivity        
        #transform them back
        #res[0] = sqrt(res[0]**2 + 1) -1
        #res[1] = sqrt(res[1]**2 + 1) -1
        
        #print res

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataROI.ravel())- len(res)))
        except Exception, e:
            pass

	#print res, fitErrors, resCode
        return GaussianFitResultR(res, self.metadata, startParameters, (xslice, yslice, zslice), resCode, fitErrors)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()
	
        x = round(x)
        y = round(y)
	
        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 0:2]
        

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
              
DESCRIPTION = 'Ratiometric multi-colour 2D Gaussian fit (no background term).'
LONG_DESCRIPTION = 'Ratiometric multi-colour 2D Gaussian fit. Uses shiftfield to correct chromatic shift during the fit. Assumes background has been subtracted and eliminates background term for better performance over SplitterFitQR.'
