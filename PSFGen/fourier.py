#!/usr/bin/python
##################
# pri.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
'''generate a phase ramp psf using fourier optics'''
from pylab import *

k = 2*pi/488 #k at 488nm

class FourierPropagator:
    def __init__(self, u,v,k):
         self.propFac = -2j*pi**2*(u**2 + v**2)/k

    def propagate(self, F, z):
        return ifftshift(ifftn(F*exp(self.propFac*z)))

def GenWidefieldAP(dx = 5):
    X, Y = meshgrid(arange(-5000, 5000., dx),arange(-5000, 5000., dx))
    u = X/(dx*dx*X.shape[0])
    v = Y/(dx*dx*X.shape[1])
    #print u.min()

    R = sqrt(u**2 + v**2)

    FP = FourierPropagator(u,v,k)

    #clf()
    #imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    M = R < 1.3/488 # NA/lambda

    return X, Y, R, FP, M

def GenWidefieldPSF(zs, dx=5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenZernikePSF(zs, dx = 5, zernikeCoeffs = []):
    from PYME.misc import zernike
    X, Y, R, FP, F = GenWidefieldAP(dx)
    
    theta = angle(X + 1j*Y)
    r = R/R[F].max()
    
    ang = 0
    
    for i, c in enumerate(zernikeCoeffs):
        ang = ang + c*zernike.zernike(i, r, theta)
        
    clf()
    imshow(angle(exp(1j*ang)))
        
    F = F.astype('d')*exp(-1j*ang)
        
    figure()
    imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenPRIPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * exp(-1j*sign(X)*.005*Y)
    clf()
    imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenAstigPSF(zs, dx=5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * exp(-1j*(.001*Y)**2)
    clf()
    imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenShiftedPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * exp(-1j*.01*Y)
    clf()
    imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenStripePRIPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * exp(-1j*sign(sin(X))*.005*Y)
    clf()
    imshow(angle(F), cmap=cm.hsv)

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
   









