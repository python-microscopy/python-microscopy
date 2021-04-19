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



#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('x', '<f4'), ('y', '<f4'),
              ('ch_data', '<u4'), ('ch_sigma', '<u4'), ('ch_background', '<u4')
              ]



class ChecksumFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=None, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma, **kwargs)

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        import zlib
        
        if not isinstance(sigma, np.ndarray):
            sigma = np.array(sigma)
            
        if not isinstance(background, np.ndarray):
            background = np.array(background)

        ch_data = zlib.crc32(data.data)
        ch_sigma = zlib.crc32(sigma.data)
        ch_background = zlib.crc32(background.data)
        
        
        #package results
        res = np.zeros(1, fresultdtype)
        res['tIndex'] = self.metadata.tIndex
        res['x'] = x
        res['y'] = y
        res['ch_sigma'] = ch_sigma
        res['ch_data'] = ch_data
        res['ch_background'] = ch_background
        return res

    


#so that fit tasks know which class to use
FitFactory = ChecksumFitFactory
#FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'CRC32 checksum of ROI'
LONG_DESCRIPTION = 'Takes a CRC32 checksum of detected ROIs'
USE_FOR = 'Debugging to ensure reproducible ROI extraction and background correction'
