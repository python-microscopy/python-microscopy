#!/usr/bin/python

##################
# astigEstimator.py
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
"""Calculates starting parameters for fitting to an astigmatic PSF. Note that this is already
somewhat more sophisticated than the entire 3D anaylsis used by 'QuickPalm' and the like as
it attempts to correct for coupling between the lateral centroid and defocus"""

from scipy.interpolate import splprep, splev
import numpy
import numpy as np
#from scipy import ndimage
#from pylab import *

splines = {}
rawMeas = {}

derivedMeas = {}

sintheta = 0
costheta = 1

def calibrate(interpolator, md, roiSize=5):
    global sintheta, costheta, calibrated
    #global zvals, dWidth
    #generate grid to evaluate function on
    X, Y, Z, safeRegion = interpolator.getCoords(md, slice(-roiSize,roiSize), slice(-roiSize,roiSize), slice(0, 2))
    #print Z, safeRegion

    if len(X.shape) > 1: #X is a matrix
        X_ = X[:, 0, 0]
        Y_ = Y[0, :, 0]
    else:
        X_ = X
        Y_ = Y

    z = numpy.arange(-1000, 1000, 10)
    ps = []

    #astigmatic PSF is not necessarily aligned to the axes
    #TODO - estimate rotation rather than requiring it as a parameter

    theta = md.getOrDefault('PSFRotation', 0)*numpy.pi/180.

    costheta = numpy.cos(theta)
    sintheta = numpy.sin(theta)

    for z0 in z:    
        d = interpolator.interp(X, Y, Z + z0)
#        if z0 % 100 == 0:
#            figure()
#            imshow(d)
        ps.append(_calcParams(d, X_, Y_))

    ps = numpy.array(ps)
    A, xp, yp, dw = ps.T
    
    #xp = xp - xp[z.size/2]
    #yp = yp - yp[z.size/2]

    rawMeas['A'] = A
    rawMeas['xp'] = xp
    rawMeas['yp'] = yp
    rawMeas['dw'] = dw

    sp, u = splprep([A], u=z, s=1)
    splines['A'] = sp

    sp, u = splprep([xp], u=z, s=1)
    splines['xp'] = sp

    sp, u = splprep([yp], u=z, s=1)
    splines['yp'] = sp

    #now for z - want this as function of dw (the difference in x & y std. deviations)
    #first look at dw as a function of z & smooth
    sp, u = splprep([dw], u=z, s=5)
    splines['dw'] = sp
    dw2 = splev(z, sp)[0] #evaluate to give smoothed dw values

    #unfortunately dw is not always monotonic - pull out the central section that is
    d_dw = numpy.diff(splev(numpy.arange(-1000, 1000, 10), sp)[0])

    #find whether gradient is +ve or negative
    sgn = numpy.sign(d_dw)

    #take all bits having the same gradient sign as the central bit
    mask = sgn == sgn[int(len(sgn)/2)]

    #make sure to only take the middle contiguous section
    segNo = np.cumsum(np.diff(mask))
    #print len(z), len(d_dw), len(mask), len(segNo)
    mask = segNo == segNo[int(len(sgn)/2)]
    
    mask = np.hstack(([False], mask, [False]))

    derivedMeas['d_dw'] = d_dw
    derivedMeas['mask'] = mask

    zm = z[mask]
    dwm = dw2[mask]
    
    #now sort the values by dw, so we can use dw as the dependant variable
    I = dwm.argsort()
    zm = zm[I]
    dwm = dwm[I]

    sp, u = splprep([zm], u=dwm, s=1)
    splines['z'] = sp




def _calcParams(data, X, Y):
    """calculates the \sigma_x - \sigma_y term used for z position estimation"""
    A = data.max() - data.min() #amplitude

    #threshold at half maximum and subtract threshold
    dr = numpy.maximum(data - data.min() - 0.5*A, 0).squeeze()
    dr = dr/dr.sum()

    x0 = (X[:,None]*dr).sum()
    y0 = (Y[None, :]*dr).sum()

    xn = (X - x0)[:,None]
    yn = (Y - y0)[None, :]

    #sig_xl = (numpy.maximum(0, x0 - X)[:,None]*dr).sum()/(drs)
    sig_xl = (numpy.maximum(0, xn*costheta - yn*sintheta)*dr).sum()
    #sig_xr = (numpy.maximum(0, X - x0)[:,None]*dr).sum()/(drs)

    #sig_yu = (numpy.maximum(0, y0 - Y)[None, :]*dr).sum()/(drs)
    sig_yu = (numpy.maximum(0, xn*sintheta + yn*costheta)*dr).sum()
    #sig_yd = (numpy.maximum(0, Y - y0)[None, :]*dr).sum()/(drs)

    return A, x0, y0, sig_xl - sig_yu


def getStartParameters(data, X, Y, Z=None):
    A, x0, y0, dw = _calcParams(data, X, Y)

    #clamp dw to valid region
    dw_ = min(max(dw, splines['z'][0][0]), splines['z'][0][-1])
    #lookup z
    z0 = max(min(splev(dw_, splines['z'])[0], 500), -500)

    #correct position & intensity estimates for z position
    A = A/max(splev(z0, splines['A'])[0], .1)
    x0 = x0 - splev(z0, splines['xp'])[0]
    y0 = y0 - splev(z0, splines['yp'])[0]

    b = data.min()

    return [A, x0, y0, -z0, b]

