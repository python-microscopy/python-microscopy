#!/usr/bin/python
##################
# pri.py
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
"""generate a phase ramp psf unp.sing fourier optics"""
import matplotlib.pyplot as plt
import numpy as np
# from pylab import ifftshift, ifftn, fftn, fftshift
from numpy.fft import ifftshift, ifftn, fftn, fftshift

k = 2*np.pi/488 #k at 488nm

class FourierPropagator:
    def __init__(self, u,v,k):
         self.propFac = -2j*np.pi**2*(u**2 + v**2)/k

    def propagate(self, F, z):
        return ifftshift(ifftn(F*np.exp(self.propFac*z)))

def GenWidefieldAP(dx = 5):
    X, Y = np.meshgrid(np.arange(-5000, 5000., dx),np.arange(-5000, 5000., dx))
    u = X/(dx*dx*X.shape[0])
    v = Y/(dx*dx*X.shape[1])
    #print u.min()

    R = np.sqrt(u**2 + v**2)

    FP = FourierPropagator(u,v,k)

    #clf()
    #plt.imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    M = R < 1.3/488 # NA/lambda

    return X, Y, R, FP, M

def GenWidefieldPSF(zs, dx=5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return np.abs(ps**2)
    
def GenZernikePSF(zs, dx = 5, zernikeCoeffs = []):
    from PYME.misc import zernike
    X, Y, R, FP, F = GenWidefieldAP(dx)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[F].max()
    
    ang = 0
    
    for i, c in enumerate(zernikeCoeffs):
        ang = ang + c*zernike.zernike(i, r, theta)
        
    plt.clf()
    plt.imshow(np.angle(np.exp(1j*ang)))
        
    F = F.astype('d')*np.exp(-1j*ang)
        
    plt.figure()
    plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return np.abs(ps**2)

def GenPRIPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * np.exp(-1j*np.sign(X)*.005*Y)
    plt.clf()
    plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return np.abs(ps**2)

def GenAstigPSF(zs, dx=5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * np.exp(-1j*(.001*Y)**2)
    plt.clf()
    plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return np.abs(ps**2)

def GenShiftedPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * np.exp(-1j*.01*Y)
    plt.clf()
    plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return np.abs(ps**2)

def GenStripePRIPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * np.exp(-1j*np.sign(np.sin(X))*.005*Y)
    plt.clf()
    plt.imshow(np.angle(F), cmap=plt.cm.hsv)

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return np.abs(ps**2)
   








