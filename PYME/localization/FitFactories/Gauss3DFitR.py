#!/usr/bin/python

##################
# Gauss3DFitR.py
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

#from pylab import *
from PYME.Analysis.PSFGen.ps_app import *
from PYME.Analysis._fithelpers import *

from .fitCommon import fmtSlicesUsed



def f_Gauss3d(p, X, Y, Z):
    """3D PSF model function with constant background - parameter vector [A, x0, y0, z0, background]"""
    A, x0, y0, z0, wxy, wz, b = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b

    #print X.shape

    return A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*wxy**2) - ((Z-z0)**2)/(2*wz**2)) + b

class GaussFitResult:
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
        vs = self.metadata.voxelsize_nm
        X = vs.x*np.mgrid[self.slicesUsed[0]]
        Y = vs.y*np.mgrid[self.slicesUsed[1]]
        Z = vs.z*np.mgrid[self.slicesUsed[2]]
        P = np.arange(0,1.01,.1)
        return f_PSF3d(self.fitResults, X, Y, Z, P, 2*np.pi/525, 1.47, 10e3)
        #pass
        
def replNoneWith1(n):
    if n is None:
        return 1
    else:
        return n


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'),('wxy', '<f4'),('wz', '<f4'), ('background', '<f4')]),
              ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'),('wxy', '<f4'),('wz', '<f4'), ('background', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def Gauss3dFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    slicesUsed = fmtSlicesUsed(slicesUsed)

    res = np.zeros(1, dtype=fresultdtype)
    
    n_params = len(fitResults)

    res['tIndex'] = metadata['tIndex']
    res['fitResults'].view('7f4')[:n_params] = fitResults

    if fitErr is None:
        res['fitError'].view('7f4')[:] = -5e3
    else:
        res['fitError'].view('7f4')[:n_params] = fitErr
        
    res['resultCode'] = resultCode
    res['slicesUsed'] = slicesUsed
        
    return res

    # return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, fmtSlicesUsed(slicesUsed))], dtype=fresultdtype)

class Gauss3dFitFactory:
    def __init__(self, data, metadata, background=None, **kwargs):
        self.data = data
        self.metadata = metadata
        self.background = background

    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice].astype('f')

        #generate grid to evaluate function on
        X, Y, Z = np.mgrid[xslice, yslice, zslice]

        vs = self.metadata.voxelsize_nm
        X = vs.x*X
        Y = vs.y*Y
        Z = vs.z*Z
        
        #figure()
        #imshow(dataROI.max(2))
        #estimate some start parameters...
        A = dataROI.max() - dataROI.min() #amplitude
        
        drc = dataROI - dataROI.min()
        drc = np.maximum(drc - drc.max()/2, 0)
        drc = drc / drc.sum()        
        
        #x0 =  X.mean()
        #y0 =  Y.mean()
        #z0 =  Z.mean()
        
        x0 = (X*drc).sum()
        y0 = (Y*drc).sum()
        z0 = (Z*drc).sum()

        #try fitting with start value above and below current position,
        #at the end take the one with loeset missfit
        startParameters = [3*A, x0, y0, z0, 100, 250, dataROI.min()]
        
        #print startParameters
        

        #print startParameters        

        #estimate errors in data
        #sigma = (4 + scipy.sqrt(2*dataROI)/2)

        #print X.shape, dataROI.shape, zslice

        sigma = np.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*np.maximum(dataROI, 1))/self.metadata['Camera.ElectronsPerCount']
        #do the fit
        #(res, resCode) = FitModel(f_gauss2d, startParameters, dataMean, X, Y)
        #print X
        #print Y
        #print Z

        #fit with start values above current position        
        (res1, cov_x, infodict, mesg1, resCode) = FitModelWeighted(f_Gauss3d, startParameters, dataROI, sigma, X, Y, Z)
        misfit = (infodict['fvec']**2).sum()

        
        #print res1, misfit
        
        #print res
        #print scipy.sqrt(diag(cov_x))
        #return GaussianFitResult(res, self.metadata, (xslice, yslice, zslice), resCode)

        fitErrors=None
        try:
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(dataROI.size- len(res1)))
        except Exception as e:
            pass
        #print 'foo'
        fr = Gauss3dFitResultR(res1, self.metadata, (xslice, yslice, zslice), resCode, fitErrors)
        #print 'fr:', fr
        return fr
        

    def FromPoint(self, x, y, z=None, roiHalfSize=8, axialHalfSize=5):
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()

        x = round(x)
        y = round(y)
        z = round(z)

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 
                    max((z - axialHalfSize), 0):min((z + axialHalfSize + 1), self.data.shape[2])]
        

FitFactory = Gauss3dFitFactory
FitResult = Gauss3dFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = '3D Gaussian fit for confocal data.'
LONG_DESCRIPTION = '3D Gaussian fit suitable for use on 3D data sets (e.g. Confocal). Not useful for PALM/STORM analysis.'
