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

import numpy as np
from . fitCommon import fmtSlicesUsed 
from . import FFBase 

#define the format of the results
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),
                              ('y0', '<f4'),
                              ('sigxl', '<f4'), 
                              ('sigxr', '<f4'),
                              ('sigyu', '<f4'),
                              ('sigyd', '<f4')]), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

#package results
def COIFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    return np.array([(metadata['tIndex'], fitResults.astype('f'), fmtSlicesUsed(slicesUsed))], dtype=fresultdtype)


class COIFitFactory(FFBase.FitFactory):
    def FromPoint(self, x, y, z=None, roiHalfSize=4, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        I = (data - background)[:,:,0]
             
        I = I - I.min() #this shouldn't strictly be necessary if background is already subtracted
   
        #subtract half max and threshold
        I = np.maximum(I - 0.5*I.max(), 0)
        
        #calculate amplitude
        A = I.sum()          

        #calculate centroid
        x0 =  (X*I).sum()/A
        y0 =  (Y*I).sum()/A

        #estimate std deviations (for Quickpalm like z determination)
        sig_xl = (np.maximum(0, x0 - X)*I).sum()/A
        sig_xr = (np.maximum(0, X - x0)*I).sum()/A

        sig_yu = (np.maximum(0, y0 - Y)*I).sum()/A
        sig_yd = (np.maximum(0, Y - y0)*I).sum()/A

        #package results
        res = np.array([A, x0, y0, sig_xl, sig_xr, sig_yu, sig_yd])
        return COIFitResultR(res, self.metadata, (xslice, yslice, zslice))


#so that fit tasks know which class to use
FitFactory = COIFitFactory
FitResult = COIFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = '2D Centroid (No Fit), with x and y width estimation.'
LONG_DESCRIPTION = '2D Centroid. Quick and dirty.'