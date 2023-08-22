#!/usr/bin/python

###############
# LinearInterpolator.py
#
# Copyright David Baddeley, 2012
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
################
from .baseInterpolator import __interpolator
#from numpy import *
import numpy as np
from scipy import ndimage
from PYME.localization.cInterp import cInterp

class CSInterpolator(__interpolator):
    def _precompute(self):
        """function which is called after model loading and can be
        overridden to allow for interpolation specific precomputations"""
         #compute the gradient of the PSF for interpolated jacobians
        self.gradX, self.gradY, self.gradZ = np.gradient(self.interpModel)
        self.gradX /= self.dx
        self.gradY /= self.dy
        self.gradZ /= self.dz
        
        self.interpModel = ndimage.spline_filter(self.interpModel).astype('f')
        self.gradX = ndimage.spline_filter(self.gradX).astype('f')
        self.gradY = ndimage.spline_filter(self.gradY).astype('f')
        self.gradZ = ndimage.spline_filter(self.gradZ).astype('f')
        
    def interp(self, X, Y, Z):
        """do actual interpolation at values given"""

        #X = atleast_1d(X)
        #Y = atleast_1d(Y)
        #Z = atleast_1d(Z)

        ox = X[0] - 0.5*self.PSF2Offset
        oy = Y[0]
        oz = Z #[0]

        #rx = (ox % self.dx)/self.dx
        #ry = (oy % self.dy)/self.dy
        #rz = (oz % self.dz)/self.dz

        #fx = floor(len(self.IntXVals)/2) + floor(ox/self.dx)
        #fy = floor(len(self.IntYVals)/2) + floor(oy/self.dy)
        #fz = floor(len(self.IntZVals)/2) + floor(oz/self.dz)

        #print fz
        #print rx, ry, rz

        xl = len(X)
        yl = len(Y)
        #zl = len(Z)
        
        #print(ox, oy, oz)
        
        #print self.interpModel.shape
        #print self.dx
        #print(self.PSF2Offset)


        r = cInterp.InterpolateCS(self.interpModel, ox, oy, oz, xl, yl, self.dx, self.dy,self.dz)
        
        return r #atleast_3d(r)
        
    def interpG(self, X, Y, Z):
        """do actual interpolation at values given"""

        #X = atleast_1d(X)
        #Y = atleast_1d(Y)
        #Z = atleast_1d(Z)

        ox = X[0] - 0.5*self.PSF2Offset
        oy = Y[0]
        oz = Z #[0]

        #rx = (ox % self.dx)/self.dx
        #ry = (oy % self.dy)/self.dy
        #rz = (oz % self.dz)/self.dz

        #fx = floor(len(self.IntXVals)/2) + floor(ox/self.dx)
        #fy = floor(len(self.IntYVals)/2) + floor(oy/self.dy)
        #fz = floor(len(self.IntZVals)/2) + floor(oz/self.dz)

        #print fz
        #print rx, ry, rz

        xl = len(X)
        yl = len(Y)
        #zl = len(Z)


        gX = -cInterp.InterpolateCS(self.gradX, ox, oy, oz, xl, yl, self.dx, self.dy,self.dz)
        gY = -cInterp.InterpolateCS(self.gradY, ox, oy, oz, xl, yl, self.dx, self.dy,self.dz)
        gZ = -cInterp.InterpolateCS(self.gradZ, ox, oy, oz, xl, yl, self.dx, self.dy,self.dz)
        
        return gX, gY, gZ #atleast_3d(r)

    def getCoords(self, metadata, xslice, yslice, zslice):
        """placeholder to be overrriden to return coordinates needed for interpolation"""
        #generate grid to evaluate function on
        vs = metadata.voxelsize_nm
        X = vs.x*np.mgrid[xslice]
        Y = vs.y*np.mgrid[yslice]
        Z = np.array([0]).astype('f')
        
        #print interpolator.shape

        #what region is 'safe' (ie we won't run out of model to interpret)
        #for these slices...
        if self.SplitPSF:
            xm = len(X)/2
            #dx = min((interpolator.shape[1] - len(Y))/2, xm) - 2
    
            ym = len(Y)/2
            dy = min((interpolator.shape[1] - len(Y))/2, ym) - 2
            dx = dy
    
            safeRegion = ((X[int(xm-dx)], X[int(xm+dx)]), (Y[int(ym-dy)], Y[int(ym+dy)]),(Z[0] + self.IntZVals[3], Z[0] + self.IntZVals[-3]))
        else:
            xm = (len(X)/2)
            dx = min(((interpolator.shape[0] - len(X))/2), xm) - 2
    
            ym = (len(Y)/2)
            dy = min(((interpolator.shape[1] - len(Y))/2), ym) - 2
    
            safeRegion = ((X[int(xm-dx)], X[int(xm+dx)]), (Y[int(ym-dy)], Y[int(ym+dy)]),(Z[0] + self.IntZVals[3], Z[0] + self.IntZVals[-3]))

        #print('safe region: %s' % repr(safeRegion))
        return X, Y, Z, safeRegion

interpolator = CSInterpolator()
        
