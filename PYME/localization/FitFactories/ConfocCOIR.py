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

#from pylab import *
import numpy
from . import fitCommon
from PYME.localization.cModels.gauss_app import *


fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),
                              ('y0', '<f4'),
                              ('sigxl', '<f4'), 
                              ('sigxr', '<f4'),
                              ('sigyu', '<f4'),
                              ('sigyd', '<f4')])]

def COIFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):

    tIndex = metadata['tIndex']

    return numpy.array([(tIndex, fitResults.astype('f'))], dtype=fresultdtype)


def ConfocCOI(data, metadata, thresh=5, background=None):
    dataROI = data.squeeze()

    #average in z
    #dataMean = dataROI.mean(2) - self.metadata.CCD.ADOffset

    #generate grid to evaluate function on
    vs = metadata.voxelsize_nm
    X, Y = numpy.mgrid[:dataROI.shape[0], :dataROI.shape[1]]
    X = vs.x*X
    Y = vs.y*Y


    if not background is None and len(numpy.shape(background)) > 1 and not ('Analysis.subtractBackground' in metadata.getEntryNames() and metadata['Analysis.subtractBackground'] == False):
        bgROI = background.squeeze()

        dataROI = dataROI - bgROI

    dataROI = (dataROI*(dataROI > thresh) - thresh).astype('f')


    A = dataROI.sum()

    #print Xg.shape, Ag.shape
    x0 =  (X*dataROI).sum()/A
    y0 =  (Y*dataROI).sum()/A

    sig_xl = (numpy.maximum(0, x0 - X)*dataROI).sum()/A
    sig_xr = (numpy.maximum(0, X - x0)*dataROI).sum()/A

    sig_yu = (numpy.maximum(0, y0 - Y)*dataROI).sum()/A
    sig_yd = (numpy.maximum(0, Y - y0)*dataROI).sum()/A

    
    res = numpy.array([A, x0, y0, sig_xl, sig_xr, sig_yu, sig_yd])

    return COIFitResultR(res, metadata)

    

#so that fit tasks know which class to use
FitFactory = ConfocCOI
FitResult = COIFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray


DESCRIPTION = '3D centroid for confocal data.'
LONG_DESCRIPTION = '3D centroid suitable for use on 3D data sets (e.g. Confocal). Not useful for PALM/STORM analysis.'
