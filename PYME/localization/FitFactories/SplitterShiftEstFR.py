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
#from pylab import *
#import copy_reg
#import numpy
import types

#import PYME.Analysis.points.twoColour as twoColour

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


def f_gauss2d2c(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR, d_x, d_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    r = genGauss(X,Y,Ar,x0,y0,s,bR,0,0)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(X,Y,Ag,x0 + d_x,y0 + d_y,s,bG,0,0)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....
    
    return np.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)


        
def replNoneWith1(n):
    if n is None:
        return 1
    else:
        return n


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),
                              ('Ar', '<f4'),
                              ('x0', '<f4'),
                              ('y0', '<f4'),
                              ('sigma', '<f4'),
                              ('background_g', '<f4'),
                              ('background_r', '<f4'),
                              ('dx', '<f4'),('dy', '<f4')]),
              ('fitError', [('Ag', '<f4'),('Ar', '<f4'),
                            ('x0', '<f4'),('y0', '<f4'),
                            ('sigma', '<f4'),
                            ('background_g', '<f4'),('background_r', '<f4'),
                            ('dx', '<f4'),('dy', '<f4')]),
              ('resultCode', '<i4'),
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    if slicesUsed is None:
        slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
    else:
        slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))

    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')

    #print slicesUsed

    #tIndex = metadata['tIndex']

    #print fitResults.dtype
    #print fitErr.dtype
    #print fitResults
    #print fitErr
    #print tIndex
    #print slicesUsed
    #print resultCode

    res = np.zeros(1, dtype=fresultdtype)

    n_params = len(fitResults)

    res['tIndex'] = metadata['tIndex']
    res['fitResults'].view('9f4')[:n_params] = fitResults

    if fitErr is None:
        res['fitError'].view('9f4')[:] = -5e3
    else:
        res['fitError'].view('9f4')[:n_params] = fitErr

    res['resultCode'] = resultCode
    res['slicesUsed'] = slicesUsed
    
    return res


class GaussianFitFactory:
    def __init__(self, data, metadata, fitfcn=f_gauss2d2c, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
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
        dataROI = self.data[xslice, yslice, zslice]


        #generate grid to evaluate function on
        vs = self.metadata.voxelsize_nm
        X = vs.x*scipy.mgrid[xslice]
        Y = vs.y*scipy.mgrid[yslice]

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata['chroma.dx'],self.metadata['chroma.dy'])

        #Xr = Xg + DeltaX
        #Yr = Yg + DeltaY

        #print DeltaX
        #print DeltaY

        #estimate some start parameters...
        Ag = dataROI[:,:,0].max() - dataROI[:,:,0].min() #amplitude
        Ar = dataROI[:,:,1].max() - dataROI[:,:,1].min() #amplitude

#        figure(2)
#        clf()
#        subplot(121)
#        imshow(dataROI[:,:,0], interpolation='nearest')
##
#        subplot(122)
#        imshow(dataROI[:,:,1], interpolation='nearest')

        #print Ag
        #print Ar

        x0 =  X.mean()
        y0 =  Y.mean()

        startParameters = [Ag, Ar, x0, y0, 250/2.35, dataROI[:,:,0].min(),dataROI[:,:,1].min(), .001, .001]

        #print startParameters

        #estimate errors in data
        nSlices = 1#dataROI.shape[2]
        
        #sigma = scipy.sqrt(self.metadata.CCD.ReadNoise**2 + (self.metadata.CCD.noiseFactor**2)*self.metadata.CCD.electronsPerCount*self.metadata.CCD.EMGain*dataROI)/self.metadata.CCD.electronsPerCount
        sigma = scipy.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*scipy.maximum(dataROI, 1)/nSlices)/self.metadata['Camera.ElectronsPerCount']

        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataROI, sigma, X, Y)

        

        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataROI.ravel())- len(res)))
        except Exception as e:
            pass

        #print res, fitErrors, resCode
        return GaussianFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors)

    def FromPoint(self, x, y, z=None, roiHalfSize=7, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()

        x = int(round(x))
        y = int(round(y))

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 0:2]
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = '2D ratiometric fitting shift calibration'
LONG_DESCRIPTION = '2D ratiometric fitting shift calibration. Used with a bead (or very bright single molecule) dataset to calibrate the chromatic shift between the two colour channels.'
USE_FOR = 'Calibrating the splitter'
