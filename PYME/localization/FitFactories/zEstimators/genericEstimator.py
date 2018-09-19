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
"""Uses machine learning to estimate starting parameters in a way that is (hopefully) both PSF agnostic and reasonable
quick.
"""

from scipy.interpolate import splprep, splev
import numpy
import numpy as np
from sklearn import svm
#from pylab import *

splines = {}
rawMeas = {}

_svr = None

sintheta = 0
costheta = 1

TWOCHANNEL = False

DEBUG_PLOTS = False

def calibrate(interpolator, md, roiSize=5):
    global sintheta, costheta, calibrated, _svr
    #global zvals, dWidth
    #generate grid to evaluate function on
    X, Y, Z, safeRegion = interpolator.getCoords(md, slice(-roiSize,roiSize), slice(-roiSize,roiSize), slice(0, 2))
    #print Z, safeRegion
    axialShift = md.getOrDefault('Analysis.AxialShift', 0)
    if axialShift is None:
        axialShift = 0
    
    ratio = 0.5#md.Analysis.ColourRatio

    if len(X.shape) > 1: #X is a matrix
        X_ = X[:, 0, 0]
        Y_ = Y[0, :, 0]
    else:
        X_ = X
        Y_ = Y

    zmin, zmax  = safeRegion[2]
    
    #print zmin, zmax
    zmin = max(zmin, zmin - axialShift)
    zmax = min(zmax, zmax - axialShift)
    
    #print zmin, zmax
    
    z = numpy.arange(max(-800, zmin), min(810, zmax), 10)
    ps = []
    
    #print axialShift
    dx = 0*70*np.random.normal(size=len(z))
    dy = 0*70 * np.random.normal(size=len(z))

    for x0, y0, z0 in zip(dx, dy, z):
        d1 = interpolator.interp(X + x0, Y+ y0, Z + z0)
        
        ps.append(_calcParams(np.random.poisson(2+ (1000 + np.random.exponential(1000))*d1), X_, Y_))
        #ps.append(_calcParams(numpy.concatenate([numpy.atleast_3d(d1), numpy.atleast_3d(d2)], 2), X_, Y_))

    ps = numpy.array(ps).T
    xp, yp =  ps[:2, :]
    
    A = ps[2]
    feat = ps[3:]
    
    #print ps, feat
    
    rawMeas['feat'] = feat
    rawMeas['z'] = z

    rawMeas['xp'] = xp + dx
    rawMeas['yp'] = yp + dy
    
    _svr = svm.SVR(C=1000)
    _svr.fit(feat.T, z)

    sp, u = splprep([xp], u=z, s=1)
    splines['xp'] = sp

    sp, u = splprep([yp], u=z, s=1)
    splines['yp'] = sp
    
    splines['z'] = True
    
    if DEBUG_PLOTS:
        #print feat.shape
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(5, 10))
        n_feats = feat.shape[0]
        for i in range(n_feats):
            plt.subplot(n_feats, 1, i+1)
            plt.plot(z, feat[i, :])
    

    
    




def _calcParams(data, X, Y):
    """calculates the \sigma_x - \sigma_y term used for z position estimation"""
    data = np.atleast_3d(data)
    A = np.atleast_1d((data - data.min(1).min(0)[None,None,:]).sum(1).sum(0)) #amplitude
    A_ = np.atleast_1d(data.max(1).max(0) - data.min(1).min(0))
    
    #print data.shape, A, X
    
    #threshold at half maximum and subtract threshold
    dr = numpy.maximum(data - data.min(1).min(0) - 0.2*A_[None,None, :], 0)#.squeeze()

    dr = numpy.maximum(data - data.mean(1).mean(0), 0)#.squeeze()
    #dr = (data - data.mean()).squeeze()
    
    #print A
    
    if len(A) == 2:
        sr = dr.sum(1).sum(0)
        sr = (sr[0]/sr.sum(), )
    else:
        sr = ()
        
    
    dr = dr/dr.sum(1).sum(0)[None,None,:]

    if False:#DEBUG_PLOTS:
        #print feat.shape
        import matplotlib.pyplot as plt
    
        plt.figure(figsize=(5, 3))
        plt.subplot(1,2,1)
        plt.imshow(dr[:,:,0])
        plt.subplot(1, 2, 2)
        plt.imshow(dr[:, :, 1])
    #print dr.sum(1).sum(0)[None,None,:] 

    x0 = (X[:,None, None]*dr).sum(1).sum(0)
    y0 = (Y[None, :, None]*dr).sum(1).sum(0)
    
    x0_ = (x0*A).sum()/A.sum()
    y0_ = (y0 * A).sum() / A.sum()
    
    
    #print x0

    xn = X[:,None, None] - x0_
    yn = Y[None, :, None] - y0_
    
    #r = numpy.sqrt(xn*xn + yn*yn)

    #sig = numpy.sqrt((xn*xn*dr + yn*yn*dr).sum(1).sum(0)) #+ (xn*xn*dr).sum(1).sum(0)

    rn2 = xn*xn + yn*yn

    #print data.shape,
    sig_x = (numpy.abs(xn) * dr).sum(1).sum(0)/80.
    sig_y = (numpy.abs(yn) * dr).sum(1).sum(0)/80.
    
    d_sig = (sig_x - sig_y)
    
    #d2_sig = (sig_x[1] - sig_x[0], sig_y[1] - sig_y[0])
    
    d1 = (rn2 < (2*70.)**2).astype('f')*data
    #print d1.shape

    sig = d1.sum(1).sum(0)/data.sum(1).sum(0)

    
    #print A.mean(), x0[0], y0[0], sig[1] - sig[0]
    return (x0_, y0_) + tuple(A/1e3) + tuple(d_sig) + tuple(0.5*(sig_x + sig_y) - 1)  #+ sr


def getStartParameters(data, X, Y, Z=None):
    p = _calcParams(data, X, Y)
    x0, y0 = p[:2]
    
    A = 1e3*np.array(p[2])
    
    feat = p[3:]
    #print 'p:', p
    #print 'feat:', feat
    
    z0 = _svr.predict(np.array(feat).reshape(1, -1))
    z0 = -max(min(z0, 800), -800)
    #print z0

    #correct position & intensity estimates for z position
    #A = A/splev(z0, splines['A'])[0]
    x0 = x0 #- splev(z0, splines['xp'])[0]
    y0 = y0 #- splev(z0, splines['yp'])[0]

    b = data.min()
    
    #print 'sp', x0, y0, z0

    return [np.mean(A), x0, y0, z0, b]

