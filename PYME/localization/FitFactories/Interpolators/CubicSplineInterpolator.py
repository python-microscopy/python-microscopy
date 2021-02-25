#!/usr/bin/python

###############
# CubicSplineInterpolator.py
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
from numpy import *
from scipy import ndimage

class CubicSplineInterpolator(__interpolator):
    def _precompute(self):
        """function which is called after model loading and can be
        overridden to allow for interpolation specific precomputations"""
         #do the spline filtering here rather than in interpolation
        self.interpModel = ndimage.spline_filter(self.interpModel)


    def interp(self, X, Y, Z):
        """do actual interpolation at values given"""

        x1 = X/self.dx + len(self.IntXVals)/2.
        y1 = Y/self.dy + len(self.IntYVals)/2.
        z1 = Z/self.dz + len(self.IntZVals)/2.

        #print x1

        return ndimage.interpolation.map_coordinates(self.interpModel, array([x1, y1, z1]), mode='nearest', prefilter=False).squeeze()

    def getCoords(self, metadata, xslice, yslice, zslice):
        """placeholder to be overrriden to return coordinates needed for interpolation"""
        #generate grid to evaluate function on
        X,Y,Z = mgrid[xslice, yslice, :1]

        vs = metadata.voxelsize_nm
        X = vs.x*X
        Y = vs.y*Y
        Z = vs.z*Z

        #what region is 'safe' (ie we won't run out of model to interpret)
        #for these slices...
        xm = X.shape[0]/2
        dx = min((interpolator.shape[0] - X.shape[0])/2, xm) - 2

        ym = X.shape[1]/2
        dy = min((interpolator.shape[1] - X.shape[1])/2, ym) - 2

        safeRegion = ((X[xm-dx, 0,0], X[xm+dx, 0, 0]), (Y[0, ym-dy, 0], Y[0, ym+dy, 0]),(Z[0,0,0] + self.IntZVals[2], Z[0,0,0] + self.IntZVals[-2]))

        return X, Y, Z, safeRegion

interpolator = CubicSplineInterpolator()
        