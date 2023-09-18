#!/usr/bin/python

##################
# LatPSFFitR.py
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

from PYME.Analysis.PSFGen.ps_app import genWidefieldPSF
from PYME.Analysis._fithelpers import FitModelWeighted

from .fitCommon import fmtSlicesUsed
from . import FFBase


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
        vs = self.metadata.voxelsize_nm
        X = vs.x*np.mgrid[self.slicesUsed[0]]
        Y = vs.y*np.mgrid[self.slicesUsed[1]]
        Z = vs.z*np.mgrid[self.slicesUsed[2]]
        P = np.arange(0,1.01,.1)
        return f_PSF3d(self.fitResults, X, Y, Z, P, 2*np.pi/525, 1.47, 10e3)
        #pass


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]),
              ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('z0', '<f4'), ('background', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def PSFFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')

    tIndex = metadata['tIndex']

    return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, fmtSlicesUsed(slicesUsed))], dtype=fresultdtype)

class PSFFitFactory(FFBase.FitFactory):
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)
        
        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        #generate grid to evaluate function on
        vs = self.metadata.voxelsize_nm
        X = vs.x*np.mgrid[xslice]
        Y = vs.y*np.mgrid[yslice]
        Z = vs.z*np.mgrid[zslice]
        P = np.arange(0,1.01,.01)

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
        sigma = np.sqrt(self.metadata['Camera.ReadNoise']**2 + (self.metadata['Camera.NoiseFactor']**2)*self.metadata['Camera.ElectronsPerCount']*self.metadata['Camera.TrueEMGain']*np.maximum(dataROI, 1))/self.metadata['Camera.ElectronsPerCount']

        #fit with start values above current position        
        (res1, cov_x1, infodict1, mesg1, resCode1) = FitModelWeighted(f_PSF3d, startParameters1, dataROI, sigma, X, Y, Z, P, 2*np.pi/488, 1.47, 50e3)
        misfit1 = (infodict1['fvec']**2).sum()

        #fit with start values below current position        
        (res2, cov_x2, infodict2, mesg2, resCode2) = FitModelWeighted(f_PSF3d, startParameters2, dataROI, sigma, X, Y, Z, P, 2*np.pi/488, 1.47, 50e3)
        misfit2 = (infodict2['fvec']**2).sum()
        
        print(('Misfit above = %f, Misfit below = %f' % (misfit1, misfit2)))
        #print res
        #print scipy.sqrt(diag(cov_x))
        #return GaussianFitResult(res, self.metadata, (xslice, yslice, zslice), resCode)
        if (misfit1 < misfit2):
            return PSFFitResultR(res1, self.metadata, (xslice, yslice, zslice), resCode1, np.sqrt(np.diag(cov_x1)))
        else:
            return PSFFitResultR(res2, self.metadata, (xslice, yslice, zslice), resCode2, np.sqrt(np.diag(cov_x2)))
        

FitFactory = PSFFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray
