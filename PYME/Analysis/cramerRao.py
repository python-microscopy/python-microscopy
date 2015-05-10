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
'''tools for estimating the Fisher information matrix and Cramer-Rao lower bound'''

from pylab import *
from scipy.special import gammaln

def lp_poisson(lam, k):
    '''log of poisson likelihood fcn'''
    return k[None,None,None,:]*log(lam)[:,:,:,None] - gammaln(k+1)[None,None,None,:] - lam[:,:,:,None]
    
def lp_poisson_n(lam, k):
    '''log of poisson likelihood fcn'''
    return k*log(lam) - gammaln(k+1) - lam

def p_poisson(lam, k):
    '''poisson likelihood fcn - calculated from log lhood for numerical stability'''
    return exp(lp_poisson(lam, k))

def CalcFisherInformZ(lam, maxK=500, voxelsize=[1,1,1]):
    k = arange(maxK).astype('f')

    lpk = lp_poisson(lam, k)
    pk = p_poisson(lam, k)

    print(('number of NaNs = %d' % isnan(pk).sum()))

    dx, dy, dz, dk = gradient(lpk)
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

    FIz = [array([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])]

    return FIz
    
def FIkz(lam, k, voxelsize):
    lpk = lp_poisson_n(lam, k)
    pk = exp(lpk)

    #print 'number of NaNs = %d' % isnan(pk).sum()

    dx, dy, dz = gradient(lpk)
    dx *= 1./voxelsize[0]
    dy *= 1./voxelsize[1]
    dz *= 1./voxelsize[2]

    Exx = (dx*dx*pk).sum(1).sum(0)
    Eyy = (dy*dy*pk).sum(1).sum(0)
    Ezz = (dz*dz*pk).sum(1).sum(0)
    Exy = (dx*dy*pk).sum(1).sum(0)
    Exz = (dx*dz*pk).sum(1).sum(0)
    Eyz = (dy*dz*pk).sum(1).sum(0)

    return array([([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])])
    
def CalcFisherInformZn(lam, maxK=500, voxelsize=[1,1,1]):
    kv = arange(maxK).astype('f')
    FIz = np.zeros([lam.shape[2],3,3])
    
    for k in kv:
        lpk = lp_poisson_n(lam, k)
        pk = exp(lpk)
    
        #print 'number of NaNs = %d' % isnan(pk).sum()
    
        dx, dy, dz = gradient(lpk)
        dx *= 1./voxelsize[0]
        dy *= 1./voxelsize[1]
        dz *= 1./voxelsize[2]
    
        Exx = (dx*dx*pk).sum(1).sum(0)
        Eyy = (dy*dy*pk).sum(1).sum(0)
        Ezz = (dz*dz*pk).sum(1).sum(0)
        Exy = (dx*dy*pk).sum(1).sum(0)
        Exz = (dx*dz*pk).sum(1).sum(0)
        Eyz = (dy*dz*pk).sum(1).sum(0)
    
        FIz += array([([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])])

    return FIz
    
def CalcFisherInformZn2(lam, maxK=500, voxelsize=[1,1,1]):
    from PYME.DSView import View3D
    lam = lam.astype('d') +  1e-2 #to prevent div/0
    fact = (1./lam)
    #print lam.max()

    #print 'number of NaNs = %d' % isnan(pk).sum()

    dx, dy, dz = gradient(lam)
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
    
    FIz = array([([[Exx[i], Exy[i], Exz[i]],[Exy[i],Eyy[i], Eyz[i]],[Exz[i],Eyz[i], Ezz[i]]]) for i in range(lam.shape[2])])

    return FIz
    

def CalcCramerReoZ(FIz):
    '''CRB is the diagonal elements of the inverse of the Fisher information matrix'''
    return array([diag(inv(FI)) for FI in FIz])












