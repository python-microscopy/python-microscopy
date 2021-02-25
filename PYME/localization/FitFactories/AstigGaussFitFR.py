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
from .fitCommon import fmtSlicesUsed, pack_results
from . import FFBase 

from PYME.localization.cModels.gauss_app import genGauss,genGaussJac, genGaussJacW
from PYME.Analysis._fithelpers import FitModelWeighted, FitModelWeightedJac


##################
# Model functions
def f_gaussAstigSlow(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sx, sy, b, b_x, b_y]"""
    A, x0, y0, sx, sy, c, b_x, b_y = p
    return A*np.exp(-(X[:,None]-x0)**2/(2*sx**2) - (Y[None,:] - y0)**2/(2*sy**2)) + c + b_x*X[:,None] + b_y*Y[None,:]
#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
                              ('sigmax', '<f4'),
                              ('sigmay', '<f4'),
                              ('background', '<f4'),
                              ('bx', '<f4'),
                              ('by', '<f4')]),
              ('fitError', [('A', '<f4'),
                            ('x0', '<f4'),
                            ('y0', '<f4'),
                            ('sigmax', '<f4'),
                            ('sigmay', '<f4'),
                            ('background', '<f4'),
                            ('bx', '<f4'),
                            ('by', '<f4')]),
              ('startParams', [('A', '<f4'),
                            ('x0', '<f4'),
                            ('y0', '<f4'),
                            ('sigmax', '<f4'),
                            ('sigmay', '<f4'),
                            ('background', '<f4'),
                            ('bx', '<f4'),
                            ('by', '<f4')]),  
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4')
              ]

# def GaussianFitResultR(fitResults, startParams, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0):
#     slicesUsed = fmtSlicesUsed(slicesUsed)
#     #print slicesUsed
#
#     if fitErr is None:
#         fitErr = -5e3*np.ones(fitResults.shape, 'f')
#
#     res = np.zeros(1, fresultdtype)
#
#     res['tIndex'] = metadata.tIndex
#     res['fitResults'].view('8f4')[0, :] = fitResults.astype('f')
#     res['fitError'].view('8f4')[0, :] = fitErr.astype('f')
#     res['resultCode'] = resultCode
#     res['slicesUsed'].view('9i4')[:] = np.array(slicesUsed, dtype='i4').ravel(),
#                                                 #dtype='i4').ravel() #fmtSlicesUsed(slicesUsed)
#     res['startParams'].view('8f4')[0, :] = startParams.astype('f')
#     #res['nchi2'] = nchi2
#     res['subtractedBackground'] = background
#
#     return res
#
#     #res =  np.array([(metadata.tIndex, fitResults.astype('f'), fitErr.astype('f'), startParams.astype('f'), resultCode, slicesUsed, background)], dtype=fresultdtype)
#     #print res
#     #return res
		

class GaussianFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_gaussAstigSlow, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma, **kwargs)

        if False:#'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else: 
            self.solver = FitModelWeighted

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background

        #estimate some start parameters...
        A = data.max() - data.min() #amplitude

        vs = self.metadata.voxelsize_nm
        x0 =  vs.x*x
        y0 =  vs.y*y
        
        bgm = np.mean(background)

        startParameters = [A, x0, y0, 250/2.35, 250/2.35, dataMean.min(), .001, .001]

        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y)

        #try to estimate errors based on the covariance matrix
        fitErrors=None
        try:       
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
            if np.any(np.isnan(fitErrors)):
                #for some reason we occasionally get negatives on the diagonal of the covariance matrix (and NaN for the fitError.
                # this shouldn't happen, but catch it here in case and flag the fit as having failed
                fitErrors = None
        except Exception:
            pass
        
        tIndex = int(self.metadata.getOrDefault('tIndex', 0))

        #package results
        #return GaussianFitResultR(res, np.array(startParameters), self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm)
        return pack_results(fresultdtype, tIndex=tIndex, fitResults=res, fitError=fitErrors,
                            startParams=np.array(startParameters),resultCode=resCode,slicesUsed=(xslice, yslice, zslice),
                            subtractedBackground=bgm)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        """Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation"""
        #generate grid to evaluate function on
        vx, vy, _ = md.voxelsize_nm
        X = vx*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = vy*np.mgrid[(y - roiHalfSize):(y + roiHalfSize + 1)]

        return (f_gaussAstigSlow(params, X, Y), X[0], Y[0], 0)


#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
#FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray


import PYME.localization.MetaDataEdit as mde

PARAMETERS = [
    mde.IntParam('Analysis.ROISize', u'ROI half size', 7),

]

DESCRIPTION = 'Vanilla 2D Gaussian fit.'
LONG_DESCRIPTION = 'Single colour 2D Gaussian fit. This should be the first stop for simple analyisis.'
USE_FOR = '2D single-colour'
