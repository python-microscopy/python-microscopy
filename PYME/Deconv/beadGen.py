#!/usr/bin/python
##################
# beadGen.py
#
# Copyright David Baddeley, 2011
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
"""Generate an 'image' of a bead of a given size to use for psf extraction"""

import numpy as np
from scipy import ndimage

def genBeadImage(radius, voxelsize):
    radius = 1.0*radius
    vx, vy, vz = voxelsize
    
    sx = np.ceil((radius - vx/2)/vx)
    sy = np.ceil((radius - vy/2)/vy)
    sz = np.ceil((radius - vz/2)/vz)

    #generate an image with 10x oversampling
    X, Y, Z = np.ogrid[(-10*sx - 5):(10*sx+6), (-10*sy-5):(10*sy+6), (-10*sz-5):(10*sz+6)]

    X = X*vx/10.
    Y = Y*vy/10.
    Z = Z*vz/10.

    #print sx, sz, X.ravel(), X.ravel()[5::10]

    bd = 1.0*((X**2 + Y**2 + Z**2) < radius**2)

    bd = ndimage.convolve(bd, np.ones([10,10,10]))

    #print bd.shape

    bd = bd[5::10, 5::10, 5::10]

    return bd/bd.sum()



