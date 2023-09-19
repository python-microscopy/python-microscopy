#!/usr/bin/python

##################
# LatObjFindFR.py
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


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]),
              ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]), 
              ('resultCode', '<i4'), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')

    tIndex = metadata['tIndex']

    return np.array([(tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, fmtSlicesUsed(slicesUsed))], dtype=fresultdtype) 
		

class GaussianFitFactory(FFBase.FitFactory):        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        if (z is None): # use position of maximum intensity
            z = self.data[x,y,:].argmax()
	
        x_ = round(x)
        y_ = round(y)
	
        xslice = slice(max((x_ - roiHalfSize), 0),min((x_ + roiHalfSize + 1),self.data.shape[0]))
        yslice = slice(max((y_ - roiHalfSize), 0),min((y_ + roiHalfSize + 1), self.data.shape[1]))
        zslice = slice(max((z - axialHalfSize), 0),min((z + axialHalfSize + 1), self.data.shape[2]))

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        #average in z
        dataMean = dataROI.mean(2)

        #estimate some start parameters...
        A = dataMean.max() - dataMean.min() #amplitude

        vs = self.metadata.voxelsize_nm
        x0 =  x*vs.x
        y0 =  y*vs.y

        startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]
        
        fitErrors=None
        
        return GaussianFitResultR(np.array(startParameters), self.metadata, (xslice, yslice, zslice), 0, fitErrors)
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'Helper function for testing object finding.'
LONG_DESCRIPTION = 'Single colour object finding test routine - used internally for testing detections. Gives ~1px accuracy.'