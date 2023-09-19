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
import numpy

from .fitCommon import fmtSlicesUsed

#import types

from PYME.localization.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *


def f_gauss1d(p, X):
    """1D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, s = p
    return A*scipy.exp(-((X-x0)**2)/(2*s**2))
    #print X.shape
    #r = genGauss(X,Y,A,x0,y0,s,b,b_x,b_y)
    #r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    #return r


#f_gauss2d.D = f_J_gauss2d


        
def replNoneWith1(n):
    if n is None:
        return 1
    else:
        return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('sigma', '<f4')]),('fitError', [('A', '<f4'),('x0', '<f4'),('sigma', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    if fitErr is None:
        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

    #print slicesUsed

    tIndex = metadata['tIndex']


    return numpy.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, fmtSlicesUsed(slicesUsed))], dtype=fresultdtype)


class GaussianFitFactory:
    def __init__(self, data, metadata, fitfcn=f_gauss1d, background=None, **kwargs):
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

    def FromPoint(self, x, y, z=None, roiHalfSize=30, axialHalfSize=15):
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        x0 = x
        y0 = y
        x = round(x)
        y = round(y)

        #return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]),
        #            max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]),
        #            max((z - axialHalfSize), 0):min((z + axialHalfSize + 1), self.data.shape[2])]

        xslice = slice(max((x - roiHalfSize), 0),min((x + roiHalfSize + 1),self.data.shape[0]))
        #yslice = slice(max((y - roiHalfSize), 0),min((y + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(max((z - axialHalfSize), 0),min((z + axialHalfSize + 1), self.data.shape[2]))

        
    #def __getitem__(self, key):
        #print key
        #xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, y, zslice]

        #average in z
        dataMean = dataROI.mean(1)

        #generate grid to evaluate function on
        vx = self.metadata.voxelsize_nm.x
        X = vx*scipy.mgrid[xslice]
        #Y = 1e3*self.metadata.voxelsize.y*scipy.mgrid[yslice]

        #estimate some start parameters...
        A = dataMean.max() - dataMean.min() #amplitude

        x0 =  vx*x0
        #y0 =  1e3*self.metadata.voxelsize.y*y0
        
        #x0 =  X.mean()
        #y0 =  Y.mean()


        #estimate errors in data
        nSlices = dataROI.shape[1]
        
        sigma = scipy.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*scipy.maximum(dataMean, 1)/nSlices)/self.metadata['Camera.ElectronsPerCount']

        if not self.background is None and len(numpy.shape(self.background)) > 1 and self.metadata.getOrDefault('Analysis.subtractBackground', True):
            bgROI = self.background[xslice, yslice, zslice]

            #average in z
            bgMean = bgROI.mean(1)
            
            dataMean = dataMean - bgMean

        startParameters = [A, x0, 1e3/2.35]


        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #(res, cov_x, infodict, mesg, resCode) = FitModelWeighted(self.fitfcn, startParameters, dataMean, sigma, X, Y)
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X)

        
        fitErrors=None
        try:       
            fitErrors = scipy.sqrt(scipy.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception as e:
            pass
        return GaussianFitResultR(res, self.metadata, (xslice, xslice, zslice), resCode, fitErrors)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        #generate grid to evaluate function on
        vs = md.voxelsize_nm
        X = vs.x*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vs.y*scipy.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return f_gauss2d(params, X, Y)


   
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
