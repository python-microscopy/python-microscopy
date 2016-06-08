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
'''generate a phase ramp psf unp.sing fourier optics'''
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import ifftshift, ifftn, fftn, fftshift

import fftw3f
from PYME.Deconv import fftwWisdom

#from scipy import ndimage

#from PYME.DSView import View3D

fftwWisdom.load_wisdom()

NTHREADS = 1
FFTWFLAGS = ['measure']

n = 1.51
lamb = 680
k = 2*np.pi*n/lamb #k at 488nm

j = np.complex64(1j)

class FourierPropagator:
    def __init__(self, u,v,k):
         self.propFac = -1j*(2*2**2*(u**2 + v**2)/k)

    def propagate(self, F, z):
        return ifftshift(ifftn(F*np.exp(self.propFac*z)))
        
class FourierPropagatorHNA:
    def __init__(self, u,v,k, lamb = 488, n=1.51):
        #print k**2
        #m = (u**2 + v**2) <= (n/lamb**2)
        #self.propFac = fftw3f.create_aligned_array(u.shape, 'complex64')
        #self.propFac = 1j*8*np.pi*np.sqrt(np.maximum((n/lamb)**2 - (u**2 + v**2), 0))
        #self.propFac = ((2*np.pi*n/lamb)*np.sqrt(np.maximum(1 - (u**2 + v**2), 0))).astype('f')
        self.propFac = ((2*np.pi*n/lamb)*np.cos(.5*np.pi*np.sqrt((u**2 + v**2)))).astype('f')
        self.pfm =(self.propFac > 0).astype('f')

        self._F = fftw3f.create_aligned_array(u.shape, 'complex64')
        self._f = fftw3f.create_aligned_array(u.shape, 'complex64')
        
        #print('Creating plans for FFTs - this might take a while')

        #calculate plans for other ffts
        self._plan_f_F = fftw3f.Plan(self._f, self._F, 'forward', flags = FFTWFLAGS, nthreads=NTHREADS)
        self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        #self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()
        
        #print('Done planning')
         #print isnan(self.propFac).sum()

    def propagate(self, F, z):
        #return ifftshift(ifftn(F*np.exp(self.propFac*z)))
        #print abs(F).sum()
        pf = self.propFac*float(z)
        fs = F*self.pfm*(np.cos(pf) + j*np.sin(pf))
        self._F[:] = fftshift(fs)
        #self._F[:] = (fs)
        self._plan_F_f()
        #print abs(self._f).sum()
        return ifftshift(self._f/np.sqrt(self._f.size))
        #return (self._f/np.sqrt(self._f.size))
        
    def propagate_r(self, f, z):
        #return ifftshift(ifftn(F*np.exp(self.propFac*z)))
        #figure()
        #plt.imshow(np.angle(f))
        self._f[:] = fftshift(f)
        self._plan_f_F()
        #figure()
        #plt.imshow(np.angle(self._F))
        pf = -self.propFac*float(z)
        return (ifftshift(self._F)*(np.cos(pf)+j*np.sin(pf)))/np.sqrt(self._f.size)
        
FourierPropagator = FourierPropagatorHNA

class FourierPropagatorClipHNA:
    def __init__(self, u,v,k, lamb = 488, n=1.51, field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0):
        #print k**2
        #m = (u**2 + v**2) <= (n/lamb**2)
        #self.propFac = fftw3f.create_aligned_array(u.shape, 'complex64')
        #self.propFac = 1j*8*np.pi*np.sqrt(np.maximum((n/lamb)**2 - (u**2 + v**2), 0))
        #self.propFac = ((2*np.pi*n/lamb)*np.sqrt(np.maximum(1 - (u**2 + v**2), 0))).astype('f')
        self.propFac = ((2*np.pi*n/lamb)*np.cos(.5*np.pi*np.sqrt((u**2 + v**2)))).astype('f')
        self.pfm =(self.propFac > 0).astype('f')
        
        #self.field_x = field_x
        #self.field_y = field_y
        self.appR = apertureNA/n
        self.apertureZGrad = apertureZGradient
        self.x = u - field_x
        self.y = v - field_y

        self._F = fftw3f.create_aligned_array(u.shape, 'complex64')
        self._f = fftw3f.create_aligned_array(u.shape, 'complex64')
        
        #print('Creating plans for FFTs - this might take a while')

        #calculate plans for other ffts
        self._plan_f_F = fftw3f.Plan(self._f, self._F, 'forward', flags = FFTWFLAGS, nthreads=NTHREADS)
        self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        #self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()
        
        #print('Done planning')
         #print isnan(self.propFac).sum()

    def propagate(self, F, z):
        #return ifftshift(ifftn(F*np.exp(self.propFac*z)))
        #print abs(F).sum()
        pf = self.propFac*float(z)
        r = max(self.appR*(1 -self.apertureZGrad*z), 0)
        #print z, r
        M = (self.x*self.x + self.y*self.y) < (r*r)
        fs = F*M*self.pfm*(np.cos(pf) + j*np.sin(pf))
        self._F[:] = fftshift(fs)
        #self._F[:] = (fs)
        self._plan_F_f()
        #print abs(self._f).sum()
        return ifftshift(self._f/np.sqrt(self._f.size))
        #return (self._f/np.sqrt(self._f.size))
        
    def propagate_r(self, f, z):
        #return ifftshift(ifftn(F*np.exp(self.propFac*z)))
        #figure()
        #plt.imshow(np.angle(f))
        self._f[:] = fftshift(f)
        self._plan_f_F()
        #figure()
        #plt.imshow(np.angle(self._F))
        pf = -self.propFac*float(z)
        return (ifftshift(self._F)*(np.cos(pf)+j*np.sin(pf)))/np.sqrt(self._f.size)


def GenWidefieldAP(dx = 5, X=None, Y=None, lamb=700, n=1.51, NA = 1.47, apodization='np.sine'):
    if X == None or Y == None:
        X, Y = np.meshgrid(np.arange(-2000, 2000., dx),np.arange(-2000, 2000., dx))
    else:
        X, Y = np.meshgrid(X,Y)
    
    X = X - X.mean()
    Y = Y - Y.mean()
        
    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    #print u.min()

    R = np.sqrt(u**2 + v**2)
    
    #print R.max()*lamb
    #print(((R/(n*lamb)).max()))
    
    #plt.imshow(R*lamb)
    #colorbar()
#    figure()
#    u_ = u[u.shape[0]/2, :]
#    plot(u_, u_)
#    plot(u_, np.sqrt(1 - u_**2))
#    plot(u_, np.sqrt(u_**2) < 1.49/2 )
#    plot(u_, np.sqrt(u_**2) < 1.49/n )
#    figure()
    
    k = 2*np.pi*n/lamb

    FP = FourierPropagator(u,v,k, lamb)

    #clf()
    #plt.imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    if apodization == None:
        M = 1.0*(R < (NA/n)) # NA/lambda
    elif apodization == 'np.sine':
        M = 1.0*(R < (NA/n))*np.sqrt(np.cos(.5*np.pi*np.minimum(R, 1)))
    
    
    
    #M = M/M.sum()
    
    #plt.imshow(M)

    return X, Y, R, FP, M, u, v
    
def GenWidefieldAPA(dx = 5, X=None, Y=None, lamb=700, n=1.51, NA = 1.47, field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0, apodizisation='np.sine'):
    if X == None or Y == None:
        X, Y = np.meshgrid(arange(-2000, 2000., dx),arange(-2000, 2000., dx))
    else:
        X, Y = np.meshgrid(X,Y)
    
    X = X - X.mean()
    Y = Y - Y.mean()
        
    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    #print u.min()

    R = np.sqrt(u**2 + v**2)
    
    #print R.max()*lamb
    #print(((R/(n*lamb)).max()))
    
    #plt.imshow(R*lamb)
    #colorbar()
#    figure()
#    u_ = u[u.shape[0]/2, :]
#    plot(u_, u_)
#    plot(u_, np.sqrt(1 - u_**2))
#    plot(u_, np.sqrt(u_**2) < 1.49/2 )
#    plot(u_, np.sqrt(u_**2) < 1.49/n )
#    figure()
    
    k = 2*np.pi*n/lamb

    FP = FourierPropagatorClipHNA(u,v,k, lamb, n, field_x, field_y, apertureNA, apertureZGradient)

    #clf()
    #plt.imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    M = 1.0*(R < (NA/n)) # NA/lambda
    
    if apodizisation == None:
        M = 1.0*(R < (NA/n)) # NA/lambda
    elif apodizisation == 'np.sine':
        M = 1.0*(R < (NA/n))*np.sqrt(np.cos(.5*np.pi*np.minimum(R, 1)))
    
    #M = M/M.sum()
    
    #plt.imshow(M)

    return X, Y, R, FP, M, u, v

def GenWidefieldPSF(zs, dx=5, lamb=700, n=1.51, NA = 1.47,apodization=None):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n=n, NA = NA, apodization=apodization)
    #figure()
    #plt.imshow(abs(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenWidefieldPSFA(zs, dx=5, lamb=700, n=1.51, NA = 1.47,field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0):
    X, Y, R, FP, F, u, v = GenWidefieldAPA(dx, lamb=lamb, n=n, NA = NA, field_x=field_x, field_y=field_y, apertureNA=apertureNA, apertureZGradient = apertureZGradient)
    #figure()
    #plt.imshow(abs(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    

def PsfFromPupil(pupil, zs, dx, lamb, apodization=None, n=1.51, NA=1.51):
    dx = float(dx)
    X, Y = np.meshgrid(dx*arange(-pupil.shape[0]/2, pupil.shape[0]/2),dx*arange(-pupil.shape[1]/2, pupil.shape[1]/2))
    print((X.min(), X.max()))
    
    X = X - X.mean()
    Y = Y - Y.mean()
    
    #print ps.shape
    #print arange(-ps.shape[0]/2, ps.shape[0]/2)

    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    
    k = 2*np.pi*n/lamb

    
    #M = 1.0*(R < (NA/(n*lamb))) # NA/lambda

    #if apodization is None:
    #    M = 1.0*(R < (NA/n)) # NA/lambda
    if apodization == 'np.sine':
        R = np.sqrt(u**2 + v**2)
        M = 1.0*(R < (NA/n))*np.sqrt(np.cos(.5*np.pi*np.minimum(R, 1)))
        pupil = pupil*M

    FP = FourierPropagator(u,v,k, lamb)
    
    ps = np.concatenate([FP.propagate(pupil, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def PsfFromPupilVect(pupil, zs, dx, lamb, shape = [61,61], apodization=None, n=1.51, NA=1.51):
    dx = float(dx)
    X, Y = np.meshgrid(dx*arange(-pupil.shape[0]/2, pupil.shape[0]/2),dx*arange(-pupil.shape[1]/2, pupil.shape[1]/2))
    print((X.min(), X.max()))
    
    X = X - X.mean()
    Y = Y - Y.mean()
    
    if shape == None:
        shape = X.shape
        
    sx = shape[0]
    sy = shape[1]
    ox = (X.shape[0] - sx)/2
    oy = (X.shape[1] - sy)/2
    ex = ox + sx
    ey = oy + sy
    
    #print ps.shape
    #print arange(-ps.shape[0]/2, ps.shape[0]/2)
    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)

    R = np.sqrt(u**2 + v**2)
    
    phi = np.angle(u+ 1j*v)
    #theta = np.arcsin(minimum(R*lamb, 1))
    theta = np.arcsin(minimum(R, 1))
    
    #figure()
    #plt.imshow(phi)
    
    #figure()
    #plt.imshow(theta)
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    k = 2*np.pi*n/lamb
    
    if apodization == 'np.sine':
        R = np.sqrt(u**2 + v**2)
        M = 1.0*(R < (NA/n))*np.sqrt(np.cos(.5*np.pi*np.minimum(R, 1)))
        pupil = pupil*M
    
    
    #M = 1.0*(R < (NA/(n*lamb))) # NA/lambda

    FP = FourierPropagator(u,v,k, lamb) 
    
    fac = ct*cp**2 + sp**2
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p = abs(ps**2)
    
    fac = (ct - 1)*cp*sp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = (ct - 1)*cp*sp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = ct*sp**2 + cp**2
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = st*cp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = st*sp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)

    return p[ox:ex, oy:ey, :] #abs(ps**2)
    
def PsfFromPupilVectFP(X, Y, R, FP, u, v, n, pupil, zs):    
    phi = np.angle(u+ 1j*v)
    theta = np.arcsin(minimum(R/n, 1))
    
    #figure()
    #plt.imshow(phi)
    
    figure()
    #plt.imshow(theta)
    print theta.min(), theta.max()
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi) 
    
    fac = ct*cp**2 + sp**2
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p = abs(ps**2)
    
    fac = (ct - 1)*cp*sp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = (ct - 1)*cp*sp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = ct*sp**2 + cp**2
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = st*cp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = st*sp
    ps = np.concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)

    return p#p[ox:ex, oy:ey, :] #abs(ps**2)

def PsfFromPupilFP(X, Y, R, FP, u, v, n, pupil, zs):
    ps = np.concatenate([FP.propagate(pupil, z)[:,:,None] for z in zs], 2)
    p = abs(ps**2)

    return p#p[ox:ex, oy:ey, :] #abs(ps**2)
   
def ExtractPupil(ps, zs, dx, lamb=488, NA=1.3, n=1.51, nIters = 50, size=5e3, intermediateUpdates=False):
    dx = float(dx)
    if not size:
        X, Y = np.meshgrid(float(dx)*arange(-ps.shape[0]/2, ps.shape[0]/2),float(dx)*arange(-ps.shape[1]/2, ps.shape[1]/2))
    else:
        X, Y = np.meshgrid(arange(-size, size, dx),arange(-size, size, dx))
        
    X = X - X.mean()
    Y = Y - Y.mean()
    
    sx = ps.shape[0]
    sy = ps.shape[1]
    ox = (X.shape[0] - sx)/2
    oy = (X.shape[1] - sy)/2
    ex = ox + sx
    ey = oy + sy
    
    #print ps.shape
    #print arange(-ps.shape[0]/2, ps.shape[0]/2)
    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    
    

    R = np.sqrt(u**2 + v**2)
    M = 1.0*(R < (NA/n)) # NA/lambda
    
    #figure()
    #plt.imshow(R)
    
    #colorbar()
    #contour(M, [0.5])
    #figure()
    
    k = 2*np.pi*n/lamb

    FP = FourierPropagator(u,v,k, lamb)

    pupil = M*np.exp(1j*0)
    
    sps = np.sqrt(ps)
    
    #normalize
    sps = np.pi*M.sum()*sps/sps.sum(1).sum(0)[None,None,:]
    
    #View3D(sps)
    
    for i in range(nIters):
        new_pupil = 0*pupil
        
        #bps = []
        #abp = []
        
        print(i)#, abs(pupil).sum()
        
        #figure()
        #plt.imshow(abs(pupil))
        res = 0

        jr = np.argsort(np.random.rand(ps.shape[2]))
                
        
        for j in jr:#range(ps.shape[2]):
            #propogate to focal plane
            prop_j  = FP.propagate(pupil, zs[j])
            
            #print abs(prop_j).sum(), np.sqrt(ps[:,:,j]).sum()
            
            #print ps[:,:,j].shape
            #print prop_j.shape
            
            #figure()
            #plt.imshow(np.angle(prop_j))
            
                        
            #print prop_j[ox:ex, oy:ey].shape, sps[:,:,j].shape
            #print abs(prop_j[ox:ex, oy:ey]).sum(), sps[:,:,j].sum()
            pj= prop_j[ox:ex, oy:ey]
            pj_mag = abs(pj)
            sps_j = sps[:,:,j]
            
            #A = np.vstack([sps_j.ravel(), np.ones_like(sps_j.ravel())]).T
            #print A.shape
            
            #x, resi, rank, s = np.linalg.lstsq(A, pj_mag.ravel())
            
            #print x, resi
            
            #res += resi
            
            #sps_j = (x[0]*sps_j - x[1]).clip(0)
            res += ((pj_mag - sps_j)**2).sum()
            
            #replace amplitude, but keep phase
            prop_j[ox:ex, oy:ey] = sps_j*np.exp(1j*np.angle(pj))
            
            #print abs(prop_j).sum()
            
            #propagate back
            bp = FP.propagate_r(prop_j, zs[j])
            
            #print abs(bp).sum()
            #figure()
            #bps.append(abs(bp))
            #abp.append(np.angle(bp))
            new_pupil += bp
            
            #figure()            
            #plt.imshow(abs(new_pupil))
            if intermediateUpdates:
                pupil = M*np.exp(1j*M*np.angle(bp))
            
        new_pupil /= ps.shape[2]
        
        #View3D(bps)
        #View3D(abp)
        
        print(('res = %f' % (res/ps.shape[2])))
        #print abs(new_pupil).sum()
        
        #np_A = abs(new_pupil)
        #np_A = ndimage.gaussian_filter(np_A,.5)
        #np_A *= M
        
        #np_P = np.angle(new_pupil)
        #np_P *= M
        #np_P = ndimage.gaussian_filter(np_P, .5)
        
        
        #plt.imshow(abs(new_pupil))
        
        #crop to NA
        #new_pupil = new_pupil*M
        
        #pupil = np_A*np.exp(1j*np_P)
        
        #pupil = new_pupil*M
        
        #only fit the phase
        pupil = M*np.exp(1j*M*np.angle(new_pupil))
    
    return pupil
    
    
    
def GenZernikePSF(zs, dx = 5, zernikeCoeffs = []):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    ang = 0
    
    for i, c in enumerate(zernikeCoeffs):
        ang = ang + c*zernike.zernike(i, r, theta)
        
    clf()
    plt.imshow(np.angle(np.exp(1j*ang)))
        
    F = F.astype('d')*np.exp(-1j*ang)
        
    figure()
    plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenZernikeDPSF(zs, dx = 5, zernikeCoeffs = {}, lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    from PYME.misc import zernike, snells
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    if ns == n:
        T = 1.0*F
    else:
        #find np.angles    
        t_t = np.minimum(r*np.arcsin(NA/n), np.pi/2)
        
        #corresponding np.angle in sample with mismatch
        t_i = snells.theta_t(t_t, n, ns)
        
        #Transmission at interface (average of S and P)
        T = 0.5*(snells.Ts(t_i, ns, n) + snells.Tp(t_i, ns, n))
        #concentration of high np.angle rays:
        T = T*F/(n*np.cos(t_t)/np.sqrt(ns*2 - (n*np.sin(t_t))**2))
    
    
    
    #plt.imshow(T*(-t_i + snells.theta_t(t_t+.001, n, ns)))
    #plt.imshow(F)
    #colorbar()
    #figure()
    #plt.imshow(T, clim=(.8, 1.2))
    #colorbar()
    #figure()
    #plt.imshow(T*(-t_i + snells.theta_t(t_t+.01, n, ns)))
    #plt.imshow(t_i - t_t)
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c*zernike.zernike(i, r, theta)
        
    #clf()
    #plt.imshow(np.angle(np.exp(1j*ang)))
        
    F = T*np.exp(-1j*ang)
        
    #figure()
    #plt.imshow(np.angle(F))
        
    if vect:
        return PsfFromPupilVectFP(X,Y,R, FP, u,v, n, F, zs)
    else:
        return PsfFromPupilFP(X,Y,R, FP, u,v, n, F, zs)

#    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)
#    
#    if beadsize == 0:
#        return abs(ps**2)
#    else:
#        p1 = abs(ps**2)
        
        
    
def GenZernikeDAPSF(zs, dx = 5, X=None, Y=None, zernikeCoeffs = {}, lamb=700, n=1.51, NA = 1.47, ns=1.51,field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0, apodizisation=None, vect=True):
    from PYME.misc import zernike, snells
    X, Y, R, FP, F, u, v = GenWidefieldAPA(dx, X = X, Y=Y, lamb=lamb, n = n, NA = NA, field_x=field_x, field_y=field_y, apertureNA=apertureNA, apertureZGradient = apertureZGradient, apodizisation=apodizisation)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    if ns == n:
        T = 1.0*F
    else:
        #find np.angles    
        t_t = np.minimum(r*np.arcsin(NA/n), np.pi/2)
        
        #corresponding np.angle in sample with mismatch
        t_i = snells.theta_t(t_t, n, ns)
        
        #Transmission at interface (average of S and P)
        T = 0.5*(snells.Ts(t_i, ns, n) + snells.Tp(t_i, ns, n))
        #concentration of high np.angle rays:
        T = T*F/(n*np.cos(t_t)/np.sqrt(ns*2 - (n*np.sin(t_t))**2))
    
    
    
    #plt.imshow(T*(-t_i + snells.theta_t(t_t+.001, n, ns)))
    #plt.imshow(F)
    #colorbar()
    #figure()
    #plt.imshow(T, clim=(.8, 1.2))
    #colorbar()
    #figure()
    #plt.imshow(T*(-t_i + snells.theta_t(t_t+.01, n, ns)))
    #plt.imshow(t_i - t_t)
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c*zernike.zernike(i, r, theta)
        
    #clf()
    #plt.imshow(np.angle(np.exp(1j*ang)))
        
    F = T*np.exp(-1j*ang)
        
    #figure()
    #plt.imshow(np.angle(F))
        
    if vect:
        return PsfFromPupilVectFP(X,Y,R, FP, u,v, n, F, zs)
    else:
        return PsfFromPupilFP(X,Y,R, FP, u,v, n, F, zs)

    #ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    #return abs(ps**2)

def GenPRIPSF(zs, dx = 5, strength=1.0, dp=0, lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    #X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA=NA)
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)

    F = F * np.exp(-1j*np.sign(X)*(10*strength*v + dp/2))
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenPRIEField(zs, dx = 5, strength=1.0, dp=0, lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    #X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA=NA)
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)

    F = F * np.exp(-1j*np.sign(X)*(10*strength*v + dp/2))
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return ps
    #return abs(ps**2)
    
def GenICPRIPSF(zs, dx = 5, strength=1.0, NA=1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA = NA)

    F = F * np.exp(-1j*np.sign(X)*10*strength*v)
    #clf()
    #plt.imshow(np.angle(F))
    
    F_ = F
    F = F*(X >= 0)

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    p1 = abs(ps**2)
    
    F = F_*(X < 0)

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return  p1 + abs(ps**2)
    
def GenColourPRIPSF(zs, dx = 5, strength=1.0, transmit = [1,1]):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)

    F = F * np.exp(-1j*np.sign(X)*10*strength*v)
    
    F = F*(np.sqrt(transmit[0])*(X < 0) +  np.sqrt(transmit[1])*(X >=0))
    #clf()
    plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenAstigPSF(zs, dx=5, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)

    F = F * np.exp(-1j*((strength*v)**2 - 0.5*(strength*R)**2))
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenSAPSF(zs, dx=5, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)

    F = F * np.exp(-1j*strength*z8)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenR3PSF(zs, dx=5, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = r**3#zernike.zernike(8, r, theta)

    F = F * np.exp(-1j*strength*z8)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenBesselPSF(zs, dx=5, rad=.95, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    #from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    #theta = np.angle(X + 1j*Y)
    
    #z8 = zernike.zernike(8, r, theta)

    F = F * (r > rad)#np.exp(-1j*strength*z8)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenSABesselPSF(zs, dx=5, rad=.95, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)

    F = F * (r > rad)*np.exp(-1j*strength*z8)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)


fps = {}
def GenSAAstigPSF(zs, dx=5, strength=1.0, SA=0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    if X == None:
        Xk = 'none'
    else:
        Xk = X.ctypes.data
    if not Xk in fps.keys():
        fpset = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
        X, Y, R, FP, F, u, v = fpset
        r = R/R[abs(F)>0].max()
        theta = np.angle(X + 1j*Y)
        
        z8 = zernike.zernike(8, r, theta)
        a_s = (v**2 - 0.5*R**2)
        
        fps[Xk] = (fpset, z8, a_s)
    else:
        fpset, z8, a_s = fps[Xk]
        X, Y, R, FP, F, u, v = fpset #GenWidefieldAP(dx, X, Y)
    

    #F = F * np.exp(-1j*(strength*a_s + SA*z8))
    pf = -(strength*a_s + SA*z8)
    F = F *(np.cos(pf) + j*np.sin(pf))
    
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
fps = {}
def GenSAPRIPSF(zs, dx=5, strength=1.0, SA=0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    Xk = X.ctypes.data
    if not Xk in fps.keys():
        fpset = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
        fps[Xk] = fpset
    else:
        fpset = fps[Xk]
    X, Y, R, FP, F, u, v = fpset #GenWidefieldAP(dx, X, Y)
    r = R/R[abs(F)>0].max()

    F = F * np.exp(-1j*np.sign(X)*10*strength*v)
    
    theta = np.angle(X + 1j*Y)
    
                
    ang = SA*zernike.zernike(8, r, theta)
            
    F = F*np.exp(-1j*ang)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenDHPSF(zs, dx=5, vortices=[0.0], lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)
    
    ph = 0*u
    
    for i, vc in enumerate(vortices):
        sgn = 1#abs(vc)*(i%2)
        ph += sgn*np.angle((u - vc) + 1j*v)

    F = F * np.exp(-1j*ph)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenCubicPhasePSF(zs, dx=5, strength=1.0, X = None, Y = None, n=1.51, NA = 1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, n = n, NA = NA)

    F = F * np.exp(-1j*strength*(u**3 + v**3))
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenShiftedPSF(zs, dx = 5):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)

    F = F * np.exp(-1j*.01*Y)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenBiplanePSF(zs, dx = 5, zshift = 500, xshift = 1, NA=1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA=NA)
    F_ = F

    F = F * np.exp(-1j*.01*xshift*Y)
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z-zshift/2)[:,:,None] for z in zs], 2)

    ps1 = abs(ps**2)
    
    F = F_ * np.exp(1j*.01*xshift*Y) 
    
    #clf()
    #plt.imshow(np.angle(F))

    ps = np.concatenate([FP.propagate(F, z+zshift/2)[:,:,None] for z in zs], 2)
    return 0.5*ps1 +0.5*abs(ps**2)

def GenStripePRIPSF(zs, dx = 5):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)

    F = F * np.exp(-1j*np.sign(np.sin(X))*.005*Y)
    #clf()
    #plt.imshow(np.angle(F), cmap=cm.hsv)

    ps = np.concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
   









