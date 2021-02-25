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


#import PYME.Analysis.points.twoColour as twoColour
from .fitCommon import fmtSlicesUsed 
from . import FFBase 


#fresultdtype=[('tIndex', '<i4'),('fitResults', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]),('fitError', [('Ag', '<f4'),('Ar', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('backgroundG', '<f4'),('backgroundR', '<f4'),('bx', '<f4'),('by', '<f4')]), ('resultCode', '<i4'), ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

fresultdtype=[('tIndex', '<i4'), ('x', '<f4'), ('y', '<f4'),
              ('data', 'f4', (11,11, 2)),
              ('sigma', 'f4', (11,11, 2)),
              ('sp', [('Xg0', '<f4'),('Yg0', '<f4'),('Xr0', '<f4'),('Yr0', '<f4')]),
              ]


class GaussianFitFactory(FFBase.FFBase):
    def __init__(self, data, metadata, fitfcn=None, background=None, noiseSigma=None, **kwargs):
        super(GaussianFitFactory, self).__init__(data, metadata, fitfcn, background, noiseSigma, **kwargs)
        
        
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        roiHalfSize=5
        Xg, Yg, Xr, Yr, dataROI, bgROI, sigma, xslice, yslice, xslice2, yslice2 = self.getSplitROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)
        
        res = np.zeros(1, fresultdtype)
        res['tIndex'] = self.metadata['tIndex']
        res['x'] = x
        res['y'] = y
        
        res['sp']['Xg0'] = Xg[0]
        res['sp']['Yg0'] = Yg[0]
        res['sp']['Xr0'] = Xr[0]
        res['sp']['Yr0'] = Yr[0]
        
        data = dataROI - bgROI
        
        res['data'][0][:data.shape[0], :data.shape[1], :] = data
        res['sigma'][0][:data.shape[0], :data.shape[1], :] = sigma
        
        return res

    
        

#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = None
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde
#from PYME.localization.FitFactories import Interpolators
#from PYME.localization.FitFactories import zEstimators

PARAMETERS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','LinearInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              #mde.FilenameParam('PSFFilename', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf'),
              mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              #mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              #mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              #mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'y', choices=['x', 'y'])
              ]
              
DESCRIPTION = 'Cut out ROI for subsequent analysis - no fitting'
LONG_DESCRIPTION = 'Cut out a ROI. useful for '
USE_FOR = '2D multi-colour'
