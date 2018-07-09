#!/usr/bin/python

##################
# twist.py
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
"""Estimates twist in a phase-ramp or double helix PSF"""

import numpy as np
import scipy.ndimage
#from pylab import *

tcAng = None
tcZ = None

def calcTwist(im, X, Y):
    an = np.mod(np.angle(X[:,None] + (1j*Y)[None, :]), np.pi)

    #imshow(an)
    
    I = np.argsort(an.ravel())

    ar = an.ravel()[I]
    imr = im.ravel()[I]

    imrs = scipy.ndimage.gaussian_filter(imr*(imr > (imr.max()/2.)), 100, 0, mode='wrap')

    #plot(ar, imrs)

    return(ar[imrs.argmax()])

def twistCal(ps, X, Y, Z):
    global tcAng, tcZ

    a = [calcTwist(ps[:,:,i],X, Y) for i in range(len(Z))]

    af = scipy.ndimage.gaussian_filter(a, 1)
    af.sort()

    tcAng = af
    tcZ = Z

def getZ(twist):
    return np.interp(twist, tcAng, tcZ)
