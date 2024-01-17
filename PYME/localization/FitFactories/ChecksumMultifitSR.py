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

#import scipy
import scipy.ndimage as ndimage
import numpy as np
from .fitCommon import fmtSlicesUsed, pack_results

from PYME.localization.cModels.gauss_app import *
from PYME.Analysis._fithelpers import *


fresultdtype=[('tIndex', '<i4'),
              ('ch_data', '<u4'), ('ch_sigma', '<u4'), ('ch_background', '<u4')
              ]


class ChecksumFitFactory:
    X = None
    Y = None
    
    def __init__(self, data, metadata, fitfcn=None, background=None, noiseSigma=None, **kwargs):
        """Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. """
        self.data = data
        self.background = background
        self.noiseSigma = noiseSigma
        self.metadata = metadata
            

    def FindAndFit(self, threshold=2, gui=False, cameraMaps = None):
        import zlib
    
        if not isinstance(self.noiseSigma, np.ndarray):
            self.noiseSigma = np.array(self.noiseSigma)
    
        if not isinstance(self.background, np.ndarray):
            self.background = np.array(self.background)
    
    
        #package results
        res = np.zeros(1, fresultdtype)
        res['tIndex'] = self.metadata['tIndex']
        res['ch_sigma'] = zlib.crc32(self.noiseSigma.data)
        res['ch_data'] = zlib.crc32(self.data.data)
        res['ch_background'] = zlib.crc32(self.background.data)
        return res
        


   
        


#this means that factory is reponsible for it's own object finding and implements
#a GetAllResults method that returns a list of localisations
MULTIFIT=True

#so that fit tasks know which class to use
FitFactory = ChecksumFitFactory
#FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'CRC32 checksum of frame'
LONG_DESCRIPTION = 'Takes a CRC32 checksum of frame'
USE_FOR = 'Debugging to ensure frame data and background are reproducible'
