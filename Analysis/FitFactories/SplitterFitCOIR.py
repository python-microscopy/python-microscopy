#!/usr/bin/python

##################
# LatGaussFitFRTC.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types

#import PYME.Analysis.twoColour as twoColour

from PYME.Analysis.cModels.gauss_app import *

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
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....
    
    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)


        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigxl', '<f4'), ('sigxr', '<f4'),('sigyu', '<f4'),('sigyd', '<f4')]), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def COIFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
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


	return numpy.array([(tIndex, fitResults.astype('f'), slicesUsed)], dtype=fresultdtype)
		

class COIFitFactory:
    def __init__(self, data, metadata, fitfcn=f_gauss2d2c, background=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in 
        metadata. '''
        self.data = data
        self.background = background
        self.metadata = metadata
        self.fitfcn = fitfcn #allow model function to be specified (to facilitate changing between accurate and fast exponential approwimations)
        if type(fitfcn) == types.FunctionType: #single function provided - use numerically estimated jacobian
            self.solver = FitModelWeighted
        else: #should be a tuple containing the fit function and its jacobian
            self.solver = FitModelWeightedJac
		
        
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


        if not self.background == None and len(numpy.shape(self.background)) > 1 and not ('Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
            bgROI = self.background[xslice, yslice, zslice] - self.metadata.Camera.ADOffset

            dataROI = dataROI - bgROI

        Ag = dataROI[:,:,0]
        Ar = dataROI[:,:,1]
        
        x0 =  (Xg*Ag + Xr*Ar).sum()/(Ag.sum() + Ar.sum())
        y0 =  (Yg*Ag + Yr*Ar).sum()/(Yg.sum() + Yr.sum())

        sig_xl = (numpy.maximum(0, x0 - Xg)*Ag + numpy.maximum(0, x0 - Xr)*Ar).sum()/(Ag.sum() + Ar.sum())
        sig_xr = (numpy.maximum(0, Xg - x0)*Ag + numpy.maximum(0, Xr - x0)*Ar).sum()/(Ag.sum() + Ar.sum())

        sig_yu = (numpy.maximum(0, y0 - Yg)*Ag + numpy.maximum(0, y0 - Yr)*Ar).sum()/(Ag.sum() + Ar.sum())
        sig_yd = (numpy.maximum(0, Yg - y0)*Ag + numpy.maximum(0, Yr - y0)*Ar).sum()/(Ag.sum() + Ar.sum())

        Ag = Ag.sum()  #amplitude
        Ar = Ag.sum()  #amplitude

	
        res = [Ag, Ar, x0, y0, sig_xl, sig_xr, sig_yu, sig_yd]
        
        return COIFitResultR(res, self.metadata, (xslice, yslice, zslice))

    def FromPoint(self, x, y, z=None, roiHalfSize=4, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()
	
        x = round(x)
        y = round(y)
	
        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 0:2]
        

#so that fit tasks know which class to use
FitFactory = COIFitFactory
FitResult = COIFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
