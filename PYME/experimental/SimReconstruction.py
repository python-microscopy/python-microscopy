# -*- coding: utf-8 -*-
import numpy as np
# from pylab import fftn, ifftn, fftshift, ifftshift
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def gMatrix(thetas):
    return np.array([[1, np.cos(th), np.sin(th)] for th in thetas]).T
    
def ReconstructAngle(imgs, angle, phases, pixelsize, period, phOffset=0):
    m = gMatrix(phases)
    mi = np.linalg.inv(m)    
    
    X, Y = np.ogrid[:imgs.shape[0], :imgs.shape[1]]
    
    X *= float(pixelsize)
    Y *= float(pixelsize)
    
    kx = np.cos(angle)*np.pi/period
    ky = np.sin(angle)*np.pi/period
    
    cps = np.dot(imgs, mi)
    
    A = cps[:,:,0]
    Br = cps[:,:,1]
    Bi = cps[:,:,2]
    
    #return A, (ifftn(fftn(A) + 2*fftn(Br*np.cos(X*kx + Y*ky)) - 2*fftn(Bi*np.sin(X*kx + Y*ky)))).real
    return A, A + 2*Br*np.cos(X*kx + Y*ky + phOffset) - 2*Bi*np.sin(X*kx + Y*ky+phOffset), Br, Bi
    
def Reconstruct(imgs, angles, phases, pixelsize, period):
    angles = np.array(angles)
    phases = np.array(phases)
    a = list(set(angles))
    
    A = 0
    B = 0
    
    for ang in a:
        idx = angles == ang
        print((idx, imgs[:,:,idx].shape, phases[idx].shape))
        A1, B1 = ReconstructAngle(imgs[:,:,idx], ang, phases[idx], pixelsize, period)
        A += A1
        B += B1
    
    return A, B
    
    

    
