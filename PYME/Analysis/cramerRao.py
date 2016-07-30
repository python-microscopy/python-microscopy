#!/usr/bin/python
##################
# cramerRao.py
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
"""tools for estimating the Fisher information matrix and Cramer-Rao lower bound"""

#from pylab import *
import numpy as np
from scipy.special import gammaln

def lp_poisson(lam, k):
    """log of poisson likelihood fcn"""
    return k[None,None,None,:]*np.log(lam)[:,:,:,None] - gammaln(k+1)[None,None,None,:] - lam[:,:,:,None]
    
def lp_poisson_n(lam, k):
    """log of poisson likelihood fcn"""
    return k*np.log(lam) - gammaln(k+1) - lam

def p_poisson(lam, k):
    """poisson likelihood fcn - calculated from log lhood for numerical stability"""
    return np.exp(lp_poisson(lam, k))

def CalcFisherInformZ(lam, maxK=500, voxelsize=[1,1,1]):
    k = np.arange(maxK).astype('f')

    lpk = lp_poisson(lam, k)
    pk = p_poisson(lam, k)

    print(('number of NaNs = %d' % isnan(pk).sum()))

    dx, dy, dz, dk = np.gradient(lpk)
    dx *= 1./voxelsize[0]
    dy *= 1./voxelsize[1]
    dz *= 1./voxelsize[2]

    del dk # we're not going to use this


    Exx = (dx*dx*pk).sum(3).sum(1).sum(0)
    Eyy = (dy*dy*pk).sum(3).sum(1).sum(0)
    Ezz = (dz*dz*pk).sum(3).sum(1).sum(0)
    Exy = (dx*dy*pk).sum(3).sum(1).sum(0)
    Exz = (dx*dz*pk).sum(3).sum(1).sum(0)
    Eyz = (dy*dz*pk).sum(3).sum(1).sum(0)

    FIz = [np.array([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])]

    return FIz
    
def FIkz(lam, k, voxelsize):
    lpk = lp_poisson_n(lam, k)
    pk = np.exp(lpk)

    #print 'number of NaNs = %d' % isnan(pk).sum()

    dx, dy, dz = np.gradient(lpk)
    dx *= 1./voxelsize[0]
    dy *= 1./voxelsize[1]
    dz *= 1./voxelsize[2]

    Exx = (dx*dx*pk).sum(1).sum(0)
    Eyy = (dy*dy*pk).sum(1).sum(0)
    Ezz = (dz*dz*pk).sum(1).sum(0)
    Exy = (dx*dy*pk).sum(1).sum(0)
    Exz = (dx*dz*pk).sum(1).sum(0)
    Eyz = (dy*dz*pk).sum(1).sum(0)

    return np.array([([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])])
    
def CalcFisherInformZn(lam, maxK=500, voxelsize=[1,1,1]):
    kv = np.arange(maxK).astype('f')
    FIz = np.zeros([lam.shape[2],3,3])
    
    for k in kv:
        lpk = lp_poisson_n(lam, k)
        pk = np.exp(lpk)
    
        #print 'number of NaNs = %d' % isnan(pk).sum()
    
        dx, dy, dz = np.gradient(lpk)
        dx *= 1./voxelsize[0]
        dy *= 1./voxelsize[1]
        dz *= 1./voxelsize[2]
    
        Exx = (dx*dx*pk).sum(1).sum(0)
        Eyy = (dy*dy*pk).sum(1).sum(0)
        Ezz = (dz*dz*pk).sum(1).sum(0)
        Exy = (dx*dy*pk).sum(1).sum(0)
        Exz = (dx*dz*pk).sum(1).sum(0)
        Eyz = (dy*dz*pk).sum(1).sum(0)
    
        FIz += np.array([([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])])

    return FIz
    
def CalcFisherInformZn2(lam, maxK=500, voxelsize=[1,1,1]):
    #from PYME.DSView import View3D
    lam = lam.astype('d') +  1e-2 #to prevent div/0
    fact = (1./lam)
    #print lam.max()

    #print 'number of NaNs = %d' % isnan(pk).sum()

    dx, dy, dz = np.gradient(lam)
    dx *= 1./voxelsize[0]
    dy *= 1./voxelsize[1]
    dz *= 1./voxelsize[2]
    
    #View3D(dx*dx*fact)

    Exx = (dx*dx*fact).sum(1).sum(0)
    Eyy = (dy*dy*fact).sum(1).sum(0)
    Ezz = (dz*dz*fact).sum(1).sum(0)
    Exy = (dx*dy*fact).sum(1).sum(0)
    Exz = (dx*dz*fact).sum(1).sum(0)
    Eyz = (dy*dz*fact).sum(1).sum(0)
    
    FIz = np.array([([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])])

    return FIz


def CalcFisherInform2D(lam, voxelsize=[1, 1]):
    """
    Calculate the Fisher Information of a 2D model

    Parameters
    ----------
    lam : ndarray
        model mean value in photoelectrons before Poisson noise process
    voxelsize : iterable
        pixel dimensions of model in nm

    Returns
    -------

    a 2x2 Fisher information matrix

    """
    #from PYME.DSView import View3D
    lam = lam.astype('d') + 1e-2 #to prevent div/0
    fact = (1. / lam)
    #print lam.max()

    #print 'number of NaNs = %d' % isnan(pk).sum()

    dx, dy = np.gradient(lam)
    dx *= 1. / voxelsize[0]
    dy *= 1. / voxelsize[1]
    #dz *= 1. / voxelsize[2]

    #View3D(dx*dx*fact)

    Exx = (dx * dx * fact).sum(1).sum(0)
    Eyy = (dy * dy * fact).sum(1).sum(0)
    #Ezz = (dz * dz * fact).sum(1).sum(0)
    Exy = (dx * dy * fact).sum(1).sum(0)
    #Exz = (dx * dz * fact).sum(1).sum(0)
    #Eyz = (dy * dz * fact).sum(1).sum(0)

    FIz = np.array(
        [([[Exx, Exy], [Exy, Eyy]]) for i in range(1)])

    return FIz
    

def CalcCramerReoZ(FIz):
    """CRB is the diagonal elements of the inverse of the Fisher information matrix"""
    return np.array([np.diag(np.linalg.inv(FI)) for FI in FIz])


def CalcFisherInfoModel(params, param_delta, modelFunc, modelargs = ()):
    """
    Calculate the Cramer-Rao bound under Poisson noise for a given model function
    (at a given point), using a numerical derivative.

    Parameters
    ----------
    params : array / list
        The parameters
    param_delta : array / list
        The amount to add to the parameters when calculating the numerical gradient
    modelFunc : function
        A function describing the model. The output should be calibrated in photo-electrons
    modelargs : iterable
        A list of additional arguments to pass to the model function

    Returns
    -------

    The NparamsxNparams Fisher information matrix

    """
    params = np.array(params)

    lam = modelFunc(params)
    fact = 1./lam

    N = len(params)

    dp_i = []
    for i, p in enumerate(params):
        p_ = params.copy()
        p_[i] = p + param_delta[i]

        dp_i.append((lam - modelFunc(p_))/param_delta[i])

    FI = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            FI[i,j] = (dp_i[i]*dp_i[j]*fact).sum()

    return FI

def CalcCramerRao(FI):
    """
    Calculate the Cramer-Rao bound for a given Fisher information matrix

    Parameters
    ----------
    FI : array
        an NparamsxNparams Fisher information matrix

    Returns
    -------

    an array of lower bounds on variances (NB the CRLB is given as a variance, rather than std. deviation).

    """
    return np.diag(np.linalg.inv(FI))











