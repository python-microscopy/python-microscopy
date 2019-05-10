#!/usr/bin/python

##################
# priEstimator.py
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
"""Calculates starting parameters for fitting to a phase ramp PSF. As it uses the
angle between the lobes it should also be able to be used for double helix PSFs
with little/no modification"""

from scipy.interpolate import splprep, splev
from scipy import ndimage
import numpy
from PYME.Analysis.binAvg import binAvg
#from pylab import *

splines = {}

rawMeas = {}

lobeSeparation = None
axis_x = None

#note that the bulk of this code is copied from astigEstimator, just replacing the
#difference in widths with a measure of rotation

def calibrate(interpolator, md, roiSize=5):
    #global zvals, dWidth
    global lobeSeparation, axis_x

        
    
    #generate grid to evaluate function on
    X, Y, Z, safeRegion = interpolator.getCoords(md, slice(-roiSize,roiSize), slice(-roiSize,roiSize), slice(0, 2))
    #print Z, safeRegion

    if len(X.shape) > 1: #X is a matrix
        X_ = X[:, 0, 0]
        Y_ = Y[0, :, 0]
    else:
        X_ = X
        Y_ = Y

    z = numpy.arange(-500, 500, 10)
    
    p = interpolator.interpModel.max(2)

    if md['PRI.Axis'] == 'x':
        I = p.max(1)
    else:
        I = p.max(0)
            
    sepr = I[int(I.size/2):].argmax()
    sepl = I[int(I.size/2)::-1].argmax()
    
    #print sepr, sepl
    
    lobeSeparation = sepr + sepl
    axis_x = md['PRI.Axis'] == 'x'    
    
    ps = []

    for z0 in z:    
        d = interpolator.interp(X, Y, Z + z0)
#        if z0 % 100 == 0:
#            figure()
#            imshow(d)
        ps.append(_calcParams(d, X_, Y_))

    ps = numpy.array(ps)
    A, xp, yp, dw = ps.T
    
    xp = xp - xp[int(z.size/2)]
    yp = yp - yp[int(z.size/2)]

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
    sp, u = splprep([dw], u=z, s=.01)
    splines['dw'] = sp
    dw2 = splev(z, sp)[0] #evaluate to give smoothed dw values

    #unfortunately dw is not always monotonic - pull out the central section that is
    d_dw = numpy.diff(splev(numpy.arange(-500, 501, 10), sp)[0])

    #find whether gradient is +ve or negative
    sgn = numpy.sign(d_dw)

    #take all bits having the same gradient sign as the central bit
    mask = sgn == sgn[int(len(sgn)/2)]

    zm = z[mask]
    dwm = dw2[mask]
    
    #now sort the values by dw, so we can use dw as the dependant variable
    I = dwm.argsort()
    zm = zm[I]
    dwm = dwm[I]

    sp, u = splprep([zm], u=dwm, s=1)
    splines['z'] = sp
    
    


def _calcParams(data, X, Y):
    """calculates the mean angle in the image, used for z position estimation"""
    A = (data.max()- data.min()) #amplitude

    #threshold at half maximum and subtract threshold
    dr = numpy.maximum(data - data.min() - 0.2*A, 0).squeeze()
    drs = dr.sum()
#    
#    labs, nlabs = ndimage.label(dr)
#    nr = 0
#    x0 = 0
#    y0 = 0
#    
#    for i in xrange(1, nlabs + 1):
#        dri = dr*(labs == i)
#        dris = dri.sum()
#        
#        if dris > A:
#            x0 += (X[:,None]*dri).sum()/dris
#            y0 += (Y[None, :]*dri).sum()/dris
#            nr += 1
    
    if len(data.shape) == 3:
        xi, yi, zi = numpy.unravel_index(data.argmax(), data.shape)
    else:
        xi, yi = numpy.unravel_index(data.argmax(), data.shape)
    
    
    if axis_x:
        if xi < data.shape[0]/2:
            x2 = min(xi + lobeSeparation, data.shape[0] - 1)
        else:
            x2 = max(0, xi - lobeSeparation)
            
        y2 = data[int(x2),:].argmax()
            
        
    else:
        x0 = X[xi]
        
        if yi < data.shape[1]/2:
            y2 = min(yi + lobeSeparation, data.shape[1] - 1)
        else:
            y2 = max(yi - lobeSeparation, 0)
            
        #print y2, lobeSeparation
            
        x2 = data[:,int(y2)].argmax()
    
#    dr = data.squeeze()[(xi - 1):(xi+2), (yi - 1):(yi +2)]
#    dr /= dr.sum()        
#    x_0 = (X[(xi - 1):(xi+2), None]*dr).sum()
#    y_0 = (Y[None, (yi - 1):(yi+2)]*dr).sum()
#    
#    dr = data.squeeze()[(x2 - 1):(x2+2), (y2 - 1):(y2 +2)]
#    dr /= dr.sum()        
#    x_2 = (X[(x2 - 1):(x2+2), None]*dr).sum()
#    y_2 = (Y[None,(y2 - 1):(y2+2)]*dr).sum()
#    
    
#    x0 = 0.5*(x_0 + x_2)
#    y0 = 0.5*(y_0 + y_2)
    
    print((xi, yi, x2, y2))
#    
    
    x0 = 0.5*(X[xi] + X[x2])
    y0 = 0.5*(Y[yi] + Y[y2])
    
    

#    print nr, x0, y0, X[xi], Y[yi]            
    
#    x0 /= nr
#    y0 /= nr
    #x0 = (X[:,None]*dr).sum()/drs
    #y0 = (Y[None, :]*dr).sum()/drs

    #sig_xl = (numpy.maximum(0, x0 - X)[:,None]*dr).sum()/(drs)
    #sig_xr = (numpy.maximum(0, X - x0)[:,None]*dr).sum()/(drs)

    #sig_yu = (numpy.maximum(0, y0 - Y)[None, :]*dr).sum()/(drs)
    #sig_yd = (numpy.maximum(0, Y - y0)[None, :]*dr).sum()/(drs)

    #mask of thresholded region
    #m = dr > 0
    
    #angle of each pixel
    theta = numpy.mod(numpy.angle((x0 -X[:,None]) +1j*(y0 - Y[None, :])), numpy.pi)

    #print theta.shape

    #import pylab
    #pylab.imshow(theta)

    thm = (theta*dr).sum()/drs
    #thm = float(numpy.mod(numpy.angle((x0 -X[xi]) +1j*(y0 - Y[yi])), numpy.pi))
    
    #A = (data - data.min()).sum()

    return A, x0, y0, thm


def getStartParameters(data, X, Y, Z=None):
    ds = ndimage.gaussian_filter(data, 1)
    A, x0, y0, dw = _calcParams(ds, X, Y)

    #clamp dw to valid region
    dw_ = min(max(dw, splines['z'][0][0]), splines['z'][0][-1])
    #lookup z
    z0 = max(min(splev(dw_, splines['z'])[0], 500), -500)

    #correct position & intensity estimates for z position
    A = A/splev(z0, splines['A'])[0]
    x0 = x0 - splev(z0, splines['xp'])[0]
    y0 = y0 - splev(z0, splines['yp'])[0]

    b = data.min()

    return [A, x0, y0, -z0, b]

