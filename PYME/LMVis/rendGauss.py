#!/usr/bin/python

##################
# rendGauss.py
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
from scipy import ndimage

from PYME.localization.cModels.gauss_app import *
import numpy as np

#def Gauss2D(Xv,Yv, A,x0,y0,s):
#    Xv = mat(Xv)
#    Yv = mat(Yv)

#    X = array(Xv.T * ones(Yv.shape)) #because mgrid doesn't want non-integral steps
#    Y = array(ones(Yv.shape).T * Yv)

#    return (A/(s*sqrt(2*pi)))*(exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)))

def Gauss2D(Xv,Yv, A,x0,y0,s):
    r = genGauss(Xv,Yv,A,x0,y0,s,0,0,0)
    r.strides = r.strides #Really dodgy hack to get around something which numpy is not doing right ....
    return r

def rendHist(xs,ys, dsShape, pixSize=1):
    return np.histogram2d(xs,ys,[np.arange(0, dsShape[0], pixSize), np.arange(0, dsShape[1], pixSize)])[0]

def rendGauss(res, X, Y, roiSize = 5, errScale = 1, cutoffErr=10, cutoffSigma = 3):
    im = np.zeros((len(X), len(Y)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = abs(X[1] - X[0]) 
    
    for r in res:
        ix = abs(X - r.x0()).argmin()
        iy = abs(Y - r.y0()).argmin()

        if (ix > roiSize) and (ix < (len(X) - roiSize)) and  (iy > roiSize) and (iy < (len(Y) - roiSize)) and  not r.fitErr is None and r.fitErr[1] < cutoffErr and r.sigma() < cutoffSigma:
            imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, r.x0(),r.y0(),max(r.fitErr[1]*errScale, delX))
            im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    return im

def rendHistF(res, X, Y, roiSize = 5, errScale = 1, cutoffErr=10, cutoffSigma = 3):
    xs = [r.x0() for r in res if not r.fitErr is None and r.fitErr[1] < cutoffErr and r.sigma() < cutoffSigma]
    ys = [r.y0() for r in res if not r.fitErr is None and r.fitErr[1] < cutoffErr and r.sigma() < cutoffSigma]

    return np.histogram2d(xs,ys,[X, Y])[0]


def rendGaussP(x,y, sx, X, Y, roiSize = 5, errScale = 1, cutoffErr=10, cutoffSigma = 3):
    im = np.zeros((len(X), len(Y)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = abs(X[1] - X[0]) 
    
    for i in range(len(x)):
        ix = abs(X - x[i]).argmin()
        iy = abs(Y - y[i]).argmin()

        if (ix > roiSize) and (ix < (len(X) - roiSize)) and  (iy > roiSize) and (iy < (len(Y) - roiSize)):
            imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, x[i],y[i],max(errScale*sx[i], delX))
            im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    return im


def rendGaussNested(res, X, Y, roiSize = 5, errScale = 1, cutoffErr=[0,100], cutoffSigma = [200/2.35, 400/2.35], cutoffA = [10, 200]):
    im = np.zeros((len(X), len(Y)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = abs(X[1] - X[0]) 
    
    for r1 in res:
        for r in r1.results:
            ix = abs(X - r.x0()).argmin()
            iy = abs(Y - r.y0()).argmin()

            if (ix > roiSize) and (ix < (len(X) - roiSize)) and  (iy > roiSize) and (iy < (len(Y) - roiSize)) and  not r.fitErr is None and r.fitErr[1] > cutoffErr[0] and r.fitErr[1] < cutoffErr[1] and r.sigma() > cutoffSigma[0]  and r.sigma() < cutoffSigma[1] and r.A() > cutoffA[0] and r.A() < cutoffA[1]:
                imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, r.x0(),r.y0(),max(r.fitErr[1]*errScale, delX))
                im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    return im



def rendGaussNestedPS(res, X, Y, roiSize = 5, errScale = 1, cutoffErr=10, cutoffZ0 = 1000, cutoffA = 0.02):
    im = np.zeros((len(X), len(Y)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = abs(X[1] - X[0]) 
    
    for r1 in res:
        for r in r1.results:
            ix = abs(X - r.x0()).argmin()
            iy = abs(Y - r.y0()).argmin()

            if (ix > roiSize) and (ix < (len(X) - roiSize)) and  (iy > roiSize) and (iy < (len(Y) - roiSize)) and  not r.fitErr is None and r.fitErr[1] < cutoffErr and abs(r.z0()) < cutoffZ0 and r.A() > cutoffA and r.resultCode==1:
                imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, r.x0(),r.y0(),max(r.fitErr[1]*errScale, delX))
                im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    return im


def rendGaussNestedXYCorr(res, X, Y, roiSize = 5, errScale = 1, cutoffErr=100, cutoffSigma = 400/2.35):
    im = np.zeros((len(X), len(Y)), 'f')

    #record our image resolution so we can plot pts with a minimum size equal to res (to avoid missing small pts)
    delX = abs(X[1] - X[0]) 
    
    for r1 in res:
        for r in r1.results:
            xg = r.metadata.voxelsize_nm.x*(r.slicesUsed[0].start + r.slicesUsed[0].stop)/2
            yg = r.metadata.voxelsize_nm.y*(r.slicesUsed[1].start + r.slicesUsed[1].stop)/2
            
            dx = r.x0() - xg
            dy = r.y0() - yg

            xn = xg + dy
            yn = yg + dx

            ix = abs(X - xn).argmin()
            iy = abs(Y - yn).argmin()

            if (ix > roiSize) and (ix < (len(X) - roiSize)) and  (iy > roiSize) and (iy < (len(Y) - roiSize)) and  not r.fitErr is None and r.fitErr[1] < cutoffErr and r.sigma() < cutoffSigma and r.resultCode==1:
                imp = Gauss2D(X[(ix - roiSize):(ix + roiSize + 1)], Y[(iy - roiSize):(iy + roiSize + 1)],1, xn,yn,max(r.fitErr[1]*errScale, delX))
                im[(ix - roiSize):(ix + roiSize + 1), (iy - roiSize):(iy + roiSize + 1)] += imp

    return im


def gaussKernel(kernel_size=21, sigma=3):
    """
    Returns a 2D Gaussian kernel
    :param kernel_size: should not be even
    :param sigma: sigma of the gaussian
    :return: array containing the gaussian values
    """
    X = np.arange(kernel_size)
    return Gauss2D(X, X, 1.0, kernel_size//2, kernel_size//2, sigma)

