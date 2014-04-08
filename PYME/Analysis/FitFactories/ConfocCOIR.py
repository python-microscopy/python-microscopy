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
#from scipy.signal import interpolate
#import scipy.ndimage as ndimage
from pylab import *
import copy_reg
import numpy
import types

#import PYME.Analysis.twoColour as twoColour

from PYME.Analysis.cModels.gauss_app import *

#from scipy import weave

from PYME.Analysis._fithelpers import *

def pickleSlice(slice):
        return unpickleSlice, (slice.start, slice.stop, slice.step)

def unpickleSlice(start, stop, step):
        return slice(start, stop, step)

copy_reg.pickle(slice, pickleSlice, unpickleSlice)


def f_gauss2d2c(p, Xg, Yg, Xr, Yr):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    Ag,Ar, x0, y0, s, bG, bR, b_x, b_y  = p
    #return A*scipy.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y
    r = genGauss(Xr,Yr,Ar,x0,y0,s,bR,b_x,b_y)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....

    g = genGauss(Xg,Yg,Ag,x0,y0,s,bG,b_x,b_y)
    g.strides = g.strides #Really dodgy hack to get around something which numpy is not doing right ....
    
    return numpy.concatenate((g.reshape(g.shape + (1,)),r.reshape(g.shape + (1,))), 2)


        
def replNoneWith1(n):
	if n == None:
		return 1
	else:
		return n


fresultdtype=[('tIndex', '<i4'),('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigxl', '<f4'), ('sigxr', '<f4'),('sigyu', '<f4'),('sigyd', '<f4')])]

def COIFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None):
#	if slicesUsed == None:
#		slicesUsed = ((-1,-1,-1),(-1,-1,-1),(-1,-1,-1))
#	else:
#		slicesUsed = ((slicesUsed[0].start,slicesUsed[0].stop,replNoneWith1(slicesUsed[0].step)),(slicesUsed[1].start,slicesUsed[1].stop,replNoneWith1(slicesUsed[1].step)),(slicesUsed[2].start,slicesUsed[2].stop,replNoneWith1(slicesUsed[2].step)))
#
#	if fitErr == None:
#		fitErr = -5e3*numpy.ones(fitResults.shape, 'f')

	#print slicesUsed

	tIndex = metadata.tIndex

	#print fitResults.dtype
	#print fitErr.dtype
	#print fitResults
	#print fitErr
	#print tIndex
	#print slicesUsed
	#print resultCode


	return numpy.array([(tIndex, fitResults.astype('f'))], dtype=fresultdtype)
		

def ConfocCOI(data, metadata, thresh=5, background=None):
    #print key
    #xslice, yslice, zslice = key

    #cut region out of data stack
    dataROI = data.squeeze() - metadata.Camera.ADOffset

    #average in z
    #dataMean = dataROI.mean(2) - self.metadata.CCD.ADOffset

    #generate grid to evaluate function on
    X, Y = scipy.mgrid[:dataROI.shape[0], :dataROI.shape[1]]
    X = 1e3*metadata.voxelsize.x*X
    Y = 1e3*metadata.voxelsize.y*Y    


    if not background == None and len(numpy.shape(background)) > 1 and not ('Analysis.subtractBackground' in metadata.getEntryNames() and metadata.Analysis.subtractBackground == False):
        bgROI = background.squeeze() - metadata.Camera.ADOffset

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

    #if A > 0:
    return COIFitResultR(res, metadata)
    #else:
    #    return []

    

#so that fit tasks know which class to use
FitFactory = ConfocCOI
FitResult = COIFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray


DESCRIPTION = '3D centroid for confocal data.'
LONG_DESCRIPTION = '3D centroid suitable for use on 3D data sets (e.g. Confocal). Not useful for PALM/STORM analysis.'