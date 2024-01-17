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

import numpy as np
from .fitCommon import fmtSlicesUsed 
from . import FFBase 

from PYME.Analysis._fithelpers import FitModelWeighted, FitModelWeightedJac

#import pylab


##################
# Model functions
def f_dumbell3d(p, X, Y, Z):
    """Pair of 3D Gaussian model functions with linear background
     - parameter vector [A, x0, y0, z0, x1, y1, z1, wxy, wz, background]
     Note: Assumes sames sigma for both Gaussian (dumbell). """
    A, x0, y0, z0, x1, y1, z1, sxy, sz, b = p
        
    r0 = A*np.exp(-((X[:,None, None]-x0)**2 + (Y[None,:,None]-y0)**2)/(2*sxy**2) - ((Z[None,None, :]-z0)**2)/(2*sz**2))
    r1 = A*np.exp(-((X[:,None, None]-x1)**2 + (Y[None,:,None]-y1)**2)/(2*sxy**2) - ((Z[None,None, :]-z1)**2)/(2*sz**2))
    r = r0+r1+b
    return r

#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),('z0', '<f4'),
                              ('x1', '<f4'),('y1', '<f4'),('z1', '<f4'),
                              ('wxy', '<f4'),('wz','<f4'),
                              ('background', '<f4')]),
              ('fitError', [('A', '<f4'),
                            ('x0', '<f4'),('y0', '<f4'),('z0', '<f4'),
                            ('x1', '<f4'),('y1', '<f4'),('z1', '<f4'),
                            ('wxy', '<f4'),('wz','<f4'), 
                            ('background', '<f4')]),
              ('length', '<f4'),
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4')
              ]

def Dumbell3DFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0, length=0):
    slicesUsed = fmtSlicesUsed(slicesUsed)
    #print slicesUsed

    # if fitErr is None:
    #     fitErr = -5e3*np.ones(fitResults.shape, 'f')
    
    # res =  np.array([(metadata.tIndex, fitResults.astype('f'), fitErr.astype('f'), length.astype('f'), resultCode, slicesUsed, background.astype('f'))], dtype=fresultdtype) 
    #print res

    res = np.zeros(1, dtype=fresultdtype)
    
    n_params = len(fitResults)
    
    res['tIndex'] = metadata['tIndex']
    res['fitResults'].view('10f4')[:n_params] = fitResults

    if fitErr is None:
        res['fitError'].view('10f4')[:] = -5e3
    else:
        res['fitError'].view('10f4')[:n_params] = fitErr
        
    res['resultCode'] = resultCode
    res['slicesUsed'] = slicesUsed
    res['subtractedBackground'] = background
        
    res['length'] = length
    return res
		

class Dumbell3DFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_dumbell3d, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma)

        self.solver = FitModelWeighted

    def FromPoint(self, x, y, z=None, roiHalfSize=7, axialHalfSize=15):
        X, Y, Z, data, background, sigma, xslice, yslice, zslice = self.get3DROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background

        #estimate some start parameters...
        A = dataMean.max() - dataMean.min() #amplitude

        vx, vy, vz = self.metadata.voxelsize_nm

        x0 = vx*x
        y0 = vy*y
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()
        z0 = vz*z
        
        bgm = np.mean(background)

        jitter_mag = 70  # TODO: Don't hardcode this?
        sxy = 250/2.35   # Guess for lateral sigma

        rand_x = jitter_mag*np.random.randn()
        rand_y = jitter_mag*np.random.randn()
        rand_z = jitter_mag*np.random.randn()

        startParameters = [3*A,             # A
                           x0+rand_x,       # x0
                           y0+rand_y,       # y0
                           z0+rand_z,       # z0
                           x0-rand_x,       # x1
                           y0-rand_y,       # y1
                           z0-rand_z,       # z1
                           sxy,             # wxy
                           2.5*sxy,         # wz
                           dataMean.min()]  # background

        # do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y, Z)

        #try to estimate errors based on the covariance matrix
        fitErrors=None
        try:       
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception:
            pass
        
        # euclidean distance between final fits
        length = np.sqrt((res[1] - res[4])**2 + (res[2] - res[5])**2 + (res[3] - res[6])**2)
        
        #package results
        return Dumbell3DFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm, length)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, z=0, roiHalfSize=7, axialHalfSize=15):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        if axialHalfSize is None:
            axialHalfSize = roiHalfSize
        #generate grid to evaluate function on
        # X = 1e3*md.voxelsize.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        # Y = 1e3*md.voxelsize.y*np.mgrid[(y - roiHalfSize):(y + roiHalfSize + 1)]
        # Z = 1e3*md.voxelsize.z*np.mgrid[(z - axialHalfSize):(z + axialHalfSize + 1)]

        xl = x-roiHalfSize
        yl = y-roiHalfSize
        zl = z-axialHalfSize

        xslice = slice(xl,x+roiHalfSize+1)
        yslice = slice(yl,y+roiHalfSize+1)
        zslice = slice(zl,z+axialHalfSize+1)

        vx, vy, vz = md.voxelsize_nm

        X = vx*np.mgrid[xslice]
        Y = vy*np.mgrid[yslice]
        Z = vz*np.mgrid[zslice]

        return (f_dumbell3d(params, X, Y, Z),vx*xl,vy*yl,vz*zl)

#so that fit tasks know which class to use
FitFactory = Dumbell3DFitFactory
FitResult = Dumbell3DFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'Fit a 3D "dumbell" consisting of 2 Gaussians'
LONG_DESCRIPTION = 'Fit a 3D "dumbell" consisting of 2 Gaussians'
USE_FOR = '3D single-colour'