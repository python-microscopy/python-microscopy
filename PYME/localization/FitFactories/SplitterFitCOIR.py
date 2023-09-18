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

import scipy
import numpy

from .fitCommon import fmtSlicesUsed
from . import FFBase



fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('Ag', '<f4'),
                              ('Ar', '<f4'),
                              ('x0', '<f4'),
                              ('y0', '<f4'),
                              ('sigxl', '<f4'), 
                              ('sigxr', '<f4'),
                              ('sigyu', '<f4'),
                              ('sigyd', '<f4')]), 
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]

def COIFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
    if fitErr is None:
        fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

    #print slicesUsed

    tIndex = metadata['tIndex']

    return numpy.array([(tIndex, fitResults.astype('f'), fmtSlicesUsed(slicesUsed))], dtype=fresultdtype)

from PYME.IO.MetaDataHandler import get_camera_roi_origin

class COIFitFactory(FFBase.FitFactory):        
    def __getitem__(self, key):
        #print key
        xslice, yslice, zslice = key

        #cut region out of data stack
        dataROI = self.data[xslice, yslice, zslice]

        #generate grid to evaluate function on
        Xg, Yg = scipy.mgrid[xslice, yslice]
        vs = self.metadata.voxelsize_nm
        Xg = vs.x*Xg
        Yg = vs.y*Yg

        #generate a corrected grid for the red channel
        #note that we're cheating a little here - for shifts which are slowly
        #varying we should be able to set Xr = Xg + delta_x(\bar{Xr}) and
        #similarly for y. For slowly varying shifts the following should be
        #equivalent to this. For rapidly varying shifts all bets are off ...

        #DeltaX, DeltaY = twoColour.getCorrection(Xg.mean(), Yg.mean(), self.metadata['chroma.dx'],self.metadata['chroma.dy'])
        roi_x0, roi_y0 = get_camera_roi_origin(self.metadata)

        x_ = Xg.mean() + roi_x0*vs.x
        y_ = Yg.mean() + roi_y0*vs.y
        DeltaX = self.metadata['chroma.dx'].ev(x_, y_)
        DeltaY = self.metadata['chroma.dy'].ev(x_, y_)

        Xr = Xg + DeltaX
        Yr = Yg + DeltaY


        if not self.background is None and len(numpy.shape(self.background)) > 1 and self.metadata.getOfDefault('Analysis.subtractBackground', True):
            bgROI = self.background[xslice, yslice, zslice]

            dataROI = dataROI - bgROI

        Ag = dataROI[:,:,0]
        Ar = dataROI[:,:,1]

        #print Xg.shape, Ag.shape
        x0 =  (Xg*Ag + Xr*Ar).sum()/(Ag.sum() + Ar.sum())
        y0 =  (Yg*Ag + Yr*Ar).sum()/(Ag.sum() + Ar.sum())

        sig_xl = (numpy.maximum(0, x0 - Xg)*Ag + numpy.maximum(0, x0 - Xr)*Ar).sum()/(Ag.sum() + Ar.sum())
        sig_xr = (numpy.maximum(0, Xg - x0)*Ag + numpy.maximum(0, Xr - x0)*Ar).sum()/(Ag.sum() + Ar.sum())

        sig_yu = (numpy.maximum(0, y0 - Yg)*Ag + numpy.maximum(0, y0 - Yr)*Ar).sum()/(Ag.sum() + Ar.sum())
        sig_yd = (numpy.maximum(0, Yg - y0)*Ag + numpy.maximum(0, Yr - y0)*Ar).sum()/(Ag.sum() + Ar.sum())

        Ag = Ag.sum()  #amplitude
        Ar = Ar.sum()  #amplitude


        res = numpy.array([Ag, Ar, x0, y0, sig_xl, sig_xr, sig_yu, sig_yd])
        
        return COIFitResultR(res, self.metadata, (xslice, yslice, zslice))

    def FromPoint(self, x, y, z=None, roiHalfSize=4, axialHalfSize=15):
        #if (z == None): # use position of maximum intensity
        #    z = self.data[x,y,:].argmax()

        x = round(x)
        y = round(y)

        return self[max((x - roiHalfSize), 0):min((x + roiHalfSize + 1),self.data.shape[0]), 
                    max((y - roiHalfSize), 0):min((y + roiHalfSize + 1), self.data.shape[1]), 0:2]
        

#so that fit tasks know which class to use
FitFactory = COIFitFactory
FitResult = COIFitResultR
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
