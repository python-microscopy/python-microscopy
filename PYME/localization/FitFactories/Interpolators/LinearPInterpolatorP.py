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
from numpy import *

class LinearInterpolator(__interpolator):
    def interp(self, X, Y, Z):
        """do actual interpolation at values given"""

        X = atleast_1d(X)
        Y = atleast_1d(Y)
        Z = atleast_1d(Z)

        ox = X[0]
        oy = Y[0]
        oz = Z[0]

        rx = (ox % self.dx)/self.dx
        ry = (oy % self.dy)/self.dy
        rz = (oz % self.dz)/self.dz

        fx = floor(len(self.IntXVals)/2) + floor(ox/self.dx)
        fy = floor(len(self.IntYVals)/2) + floor(oy/self.dy)
        fz = floor(len(self.IntZVals)/2) + floor(oz/self.dz)

        #print fz, oz
        #print rx, ry, rz

        xl = len(X)
        yl = len(Y)
        zl = len(Z)

        #print xl

        m000 = self.interpModel[fx:(fx+xl),fy:(fy+yl),fz:(fz+zl)]
        m100 = self.interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),fz:(fz+zl)]
        m010 = self.interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),fz:(fz+zl)]
        m110 = self.interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),fz:(fz+zl)]

        m001 = self.interpModel[fx:(fx+xl),fy:(fy+yl),(fz+1):(fz+zl+1)]
        m101 = self.interpModel[(fx+1):(fx+xl+1),fy:(fy+yl),(fz+1):(fz+zl+1)]
        m011 = self.interpModel[fx:(fx+xl),(fy + 1):(fy+yl+1),(fz+1):(fz+zl+1)]
        m111 = self.interpModel[(fx+1):(fx+xl+1),(fy+1):(fy+yl+1),(fz+1):(fz+zl+1)]

        #print m000.shape

    #    m = scipy.sum([((1-rx)*(1-ry)*(1-rz))*m000, ((rx)*(1-ry)*(1-rz))*m100, ((1-rx)*(ry)*(1-rz))*m010, ((rx)*(ry)*(1-rz))*m110,
    #        ((1-rx)*(1-ry)*(rz))*m001, ((rx)*(1-ry)*(rz))*m101, ((1-rx)*(ry)*(rz))*m011, ((rx)*(ry)*(rz))*m111], 0)

        m = ((1-rx)*(1-ry)*(1-rz))*m000 + ((rx)*(1-ry)*(1-rz))*m100 + ((1-rx)*(ry)*(1-rz))*m010 + ((rx)*(ry)*(1-rz))*m110+((1-rx)*(1-ry)*(rz))*m001+ ((rx)*(1-ry)*(rz))*m101+ ((1-rx)*(ry)*(rz))*m011+ ((rx)*(ry)*(rz))*m111
        #print m.shape
        return m

    def getCoords(self, metadata, xslice, yslice, zslice):
        """placeholder to be overrriden to return coordinates needed for interpolation"""
        #generate grid to evaluate function on
        vs = metadata.voxelsize_nm
        X = vs.x*mgrid[xslice]
        Y = vs.y*mgrid[yslice]
        Z = array([0]).astype('f')

        #what region is 'safe' (ie we won't run out of model to interpret)
        #for these slices...
        xm = len(X)/2
        dx = min((interpolator.shape[0] - len(X))/2, xm) - 2

        ym = len(Y)/2
        dy = min((interpolator.shape[1] - len(Y))/2, ym) - 2

        safeRegion = ((X[xm-dx], X[xm+dx]), (Y[ym-dy], Y[ym+dy]),(Z[0] + self.IntZVals[2], Z[0] + self.IntZVals[-2]))

        return X, Y, Z, safeRegion

interpolator = LinearInterpolator()
        