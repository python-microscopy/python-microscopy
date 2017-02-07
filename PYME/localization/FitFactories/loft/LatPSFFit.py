#!/usr/bin/python

##################
# LatPSFFit.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy
from scipy.signal import interpolate
import scipy.ndimage as ndimage
from pylab import *
from PYME.Analysis.PSFGen.ps_app import *

from _fithelpers import *

# import copy_reg
#
# def pickleSlice(slice):
#         return unpickleSlice, (slice.start, slice.stop, slice.step)
#
# def unpickleSlice(start, stop, step):
#         return slice(start, stop, step)
#
# copy_reg.pickle(slice, pickleSlice, unpickleSlice)



def f_PSF3d(p, X, Y, Z, P, *args):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b
    return genWidefieldPSF(X, Y, Z, P,A*1e3, x0, y0, z0, *args) + b

class PSFFitResult:
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

    def z0(self):
        return self.fitResults[3]

    def background(self):
        return self.fitResults[4]

    def renderFit(self):
        #X,Y = scipy.mgrid[self.slicesUsed[0], self.slicesUsed[1]]
        #return f_gauss2d(self.fitResults, X, Y)
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[self.slicesUsed[0]]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[self.slicesUsed[1]]
        Z = 1e3*self.metadata.voxelsize.z*scipy.mgrid[self.slicesUsed[2]]
        P = scipy.arange(0,1.01,.1)
        return f_PSF3d(self.fitResults, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)
        #pass
        


class PSFFitFactory:
    def __init__(self, data, metadata):
        self.data = data - metadata.CCD.ADOffset
        self.metadata = metadata

    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        #generate grid to evaluate function on
        X = 1e3*self.metadata.voxelsize.x*scipy.mgrid[xslice]
        Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]
        Z = 1e3*self.metadata.voxelsize.z*scipy.mgrid[zslice]
        P = scipy.arange(0,1.01,.01)

        #imshow(dataROI[:,:,0])
        #estimate some start parameters...
        A = dataROI.max() - dataROI.min() #amplitude
        x0 =  X.mean()
        y0 =  Y.mean()
        z0 =  Z.mean()

        #try fitting with start value above and below current position,
        #at the end take the one with loeset missfit
        startParameters1 = [3*A, x0, y0, z0 - 500, dataROI.min()]
        startParameters2 = [3*A, x0, y0, z0 + 500, dataROI.min()]

        #print startParameters        

        #estimate errors in data
        #sigma = (4 + scipy.sqrt(2*dataROI)/2)
        sigma = scipy.sqrt(self.metadata.CCD.ReadNoise**2 + (self.metadata.CCD.noiseFactor**2)*self.metadata.CCD.electronsPerCount*dataROI)/self.metadata.CCD.electronsPerCount
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #print X
        #print Y
        #print Z

        #fit with start values above current position        
        (res1, cov_x1, infodict1, mesg1, resCode1) = FitModelWeighted(f_PSF3d, startParameters1, dataROI, sigma, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)
        misfit1 = (infodict1['fvec']**2).sum()

        #fit with start values below current position        
        (res2, cov_x2, infodict2, mesg2, resCode2) = FitModelWeighted(f_PSF3d, startParameters2, dataROI, sigma, X, Y, Z, P, 2*scipy.pi/525, 1.47, 10e3)
        misfit2 = (infodict2['fvec']**2).sum()
        
        print(('Misfit above = %f, Misfit below = %f' % (misfit1, misfit2)))
        #print res
        #print scipy.sqrt(diag(cov_x))
        #return GaussianFitResult(res, self.metadata, (xslice, yslice, zslice), resCode)
        if (misfit1 < misfit2):
            return PSFFitResult(res1, self.metadata, (xslice, yslice, zslice), resCode1, scipy.sqrt(diag(cov_x1)))
        else:
            return PSFFitResult(res2, self.metadata, (xslice, yslice, zslice), resCode2, scipy.sqrt(diag(cov_x2)))

    def FromPoint(self, x, y, z=None, roiHalfSize=8, axialHalfSize=5):
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 
                    max((z - axialHalfSize), 0):min((z + axialHalfSize + 1), self.data.shape[2])]
        

FitFactory = PSFFitFactory
