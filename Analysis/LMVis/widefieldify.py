#!/usr/bin/python
##################
# widefieldify.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np
import scipy as sp
from scipy.fftpack import fftn, ifftn, ifftshift

from PYME.PSFGen import genWidefieldPSF

def widefieldify(image, voxelsize, wavelength=680, maxPhotonNum = 1000, backgroundLevel=.2):
    X = np.arange(image.shape[0])*voxelsize[0]
    Y = np.arange(image.shape[1])*voxelsize[1]
    Z = np.arange(image.shape[2])*voxelsize[2]
    P = np.arange(0, 1.1, .1)

    X = X - X.mean()
    Y = Y - Y.mean()
    Z = Z - Z.mean()

    PSF = genWidefieldPSF(X,Y,Z,P, k=2*np.pi/wavelength)
    PSF = PSF/PSF.sum()

    im2 = ifftshift(ifftn(fftn(PSF)*fftn(image))).real
    
    im2 = (im2 / im2.max() + backgroundLevel) * maxPhotonNum

    return np.random.poisson(im2)



    





