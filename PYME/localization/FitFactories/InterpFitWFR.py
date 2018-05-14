#!/usr/bin/python

##################
# PsfFitIR.py
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
import numpy as np



#from PYME.Analysis._fithelpers import *
#from PYME.localization.FitFactories.zEstimators import astigEstimator
#from .fitCommon import fmtSlicesUsed 
#from . import FFBase 

from .InterpFitR import PSFFitFactory, PSFFitResultR, fresultdtype, genFitImage, getDataErrors

		

class WFPSFFitFactory(PSFFitFactory):
    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, dataMean, bgMean, sigma, xslice, yslice, zslice = self.getROIAtPoint(x,y,z,roiHalfSize, axialHalfSize)       
        dataROI = dataMean - bgMean
        
        #generate grid to evaluate function on        
        X, Y, Z, safeRegion = self.interpolator.getCoords(self.metadata, xslice, yslice, zslice)

        #estimate start parameters        

        drMin = dataROI.min()
        A = 5000*dataROI.max() - drMin
        
        x0 = X.mean()
        y0 = Y.mean()
        startParameters0 = [A, x0, y0, -400, drMin]
        startParameters1 = [A, x0, y0, 400, drMin]
        
        #print startParameters0

        #do the fit
        (res0, cov_x0, infodict0, mesg0, resCode0) = self.solver(self.fitfcn, startParameters0, dataROI, sigma, self.interpolator, X, Y, Z, safeRegion)        
        (res1, cov_x1, infodict1, mesg1, resCode1) = self.solver(self.fitfcn, startParameters1, dataROI, sigma, self.interpolator, X, Y, Z, safeRegion)
        
        #normalised Chi-squared
        nchi20 = (infodict0['fvec']**2).sum()/(dataROI.size - res0.size)        
        nchi21 = (infodict1['fvec']**2).sum()/(dataROI.size - res1.size)
        
        if nchi20 < nchi21:
            fitErrors=None
            try:
                fitErrors = np.sqrt(np.diag(cov_x0) * (infodict0['fvec'] * infodict0['fvec']).sum() / (len(dataROI.ravel())- len(res0)))
            except Exception:
                pass
    
            #print res, fitErrors, resCode
            #return PSFFitResultR(res, self.metadata, numpy.array((sig_xl, sig_xr, sig_yu, sig_yd)),(xslice, yslice, zslice), resCode, fitErrors, numpy.array(startParameters), nchi2)
            return PSFFitResultR(res0, self.metadata,(xslice, yslice, zslice), resCode0, fitErrors, np.array(startParameters0), nchi20)
        else:
            fitErrors=None
            try:
                fitErrors = np.sqrt(np.diag(cov_x1) * (infodict1['fvec'] * infodict1['fvec']).sum() / (len(dataROI.ravel())- len(res1)))
            except Exception:
                pass
    
            #print res, fitErrors, resCode
            #return PSFFitResultR(res, self.metadata, numpy.array((sig_xl, sig_xr, sig_yu, sig_yd)),(xslice, yslice, zslice), resCode, fitErrors, numpy.array(startParameters), nchi2)
            return PSFFitResultR(res1, self.metadata,(xslice, yslice, zslice), resCode1, fitErrors, np.array(startParameters1), nchi21)

     

#so that fit tasks know which class to use
FitFactory = WFPSFFitFactory
FitResult = PSFFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

import PYME.localization.MetaDataEdit as mde
from PYME.localization.FitFactories import Interpolators
#from PYME.localization.FitFactories import zEstimators

#set of parameters that this fit needs to know about
PARAMETERS = [#mde.ChoiceParam('Analysis.InterpModule','Interp:','CSInterpolator', choices=Interpolators.interpolatorList, choiceNames=Interpolators.interpolatorDisplayList),
              mde.FilenameParam('PSFFile', 'PSF:', prompt='Please select PSF to use ...', wildcard='PSF Files|*.psf|TIFF files|*.tif'),
              #mde.ShiftFieldParam('chroma.ShiftFilename', 'Shifts:', prompt='Please select shiftfield to use', wildcard='Shiftfields|*.sf'),
              #mde.IntParam('Analysis.DebounceRadius', 'Debounce r:', 4),
              #mde.FloatParam('Analysis.AxialShift', 'Z Shift [nm]:', 0),
              #mde.ChoiceParam('Analysis.EstimatorModule', 'Z Start Est:', 'astigEstimator', choices=zEstimators.estimatorList),
              #mde.ChoiceParam('PRI.Axis', 'PRI Axis:', 'y', choices=['x', 'y'])
              ]
              

DESCRIPTION = 'EXP. 3D, single colour fitting for widefield PSF.'
LONG_DESCRIPTION = 'Experimental 3D, single colour fitting optimized for widefield PSF. Attempts to work around the symmetry problem by starting the fit both above and below the focus, and seeing which converges to the better fit. Uses an interpolated experimental PSF like the other Interp fits. Unlikely to work very well unless there are significant abberations (e.g. S.A.) in the PSF.'
