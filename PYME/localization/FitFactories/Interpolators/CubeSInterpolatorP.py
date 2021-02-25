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
from scipy import ndimage
from numpy import *

def _splcoefs(r):
    #st = -1
    st = float(int(floor(r)) - 1)#3./2)
    coeffs = zeros(4)
    
    for hh in range(4):
        y = abs(st - r + hh)
        
        if (y < 1.0):
            coeffs[hh] = (y*y*(y-2.0)*3.0 + 4.)/6.
        elif (y < 2.):
            y = 2.0 - y
            coeffs[hh] = y*y*y/6.
            
    return coeffs
            

class CSInterpolator(__interpolator):
    def _precompute(self):
        """function which is called after model loading and can be
        overridden to allow for interpolation specific precomputations"""
         #do the spline filtering here rather than in interpolation
        self.interpModel = ndimage.spline_filter(self.interpModel)
        
   
            
            
        
    def interp(self, X, Y, Z):
        """do actual interpolation at values given"""

        X = atleast_1d(X) #ox = X[0] 
        Y = atleast_1d(Y)
        Z = atleast_1d(Z)

        ox = X[0] - 0.5*self.PSF2Offset
        oy = Y[0]
        oz = Z[0]

        rx = ((ox + 973*self.dx) % self.dx)/self.dx #- .5
        ry = ((oy  + 973*self.dy) % self.dy)/self.dy #- .5
        rz = ((oz  + 973*self.dz) % self.dz)/self.dz #- .5

        fx = floor(len(self.IntXVals)/2) + floor(ox/self.dx)
        fy = floor(len(self.IntYVals)/2) + floor(oy/self.dy)
        fz = floor(len(self.IntZVals)/2) + floor(oz/self.dz)

        #print fz, oz
        #print rx, ry, rz
        cx = _splcoefs(rx)
        cy = _splcoefs(ry)
        cz = _splcoefs(rz)
        
        #print rx, ry, rz, cz

        xl = len(X)
        yl = len(Y)
        zl = len(Z)
        
        def _ms(xs, ys, zs):
            return self.interpModel[(fx+xs):(fx+xl+xs),(fy+xs):(fy+yl+xs),(fz+zs):(fz+zl+zs)]

        #print xl

        m = 0
        
        for xs in range(4):
            for ys in range(4):
                for zs in range(4):
                    m += cx[xs]*cy[ys]*cz[zs]*_ms(xs-1, ys-1, zs-1)
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

interpolator = CSInterpolator()
        