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
'''generate a phase ramp psf using fourier optics'''
from pylab import *
import fftw3f
from PYME.Deconv import fftwWisdom
from scipy import ndimage

from PYME.DSView import View3D

fftwWisdom.load_wisdom()

NTHREADS = 2
FFTWFLAGS = ['measure']

n = 1.51
lamb = 680
k = 2*pi*n/lamb #k at 488nm

class FourierPropagator:
    def __init__(self, u,v,k):
         self.propFac = -1j*(2*2**2*(u**2 + v**2)/k)

    def propagate(self, F, z):
        return ifftshift(ifftn(F*exp(self.propFac*z)))
        
class FourierPropagatorHNA:
    def __init__(self, u,v,k, lamb = 488, n=1.51):
        #print k**2
        #m = (u**2 + v**2) <= (n/lamb**2)
        #self.propFac = fftw3f.create_aligned_array(u.shape, 'complex64')
        self.propFac = 1j*4*pi*sqrt(np.maximum((n/lamb)**2 - (u**2 + v**2), 0))

        self._F = fftw3f.create_aligned_array(u.shape, 'complex64')
        self._f = fftw3f.create_aligned_array(u.shape, 'complex64')
        
        print 'Creating plans for FFTs - this might take a while'

        #calculate plans for other ffts
        self._plan_f_F = fftw3f.Plan(self._f, self._F, 'forward', flags = FFTWFLAGS, nthreads=NTHREADS)
        self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        #self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()
        
        print 'Done planning'
         #print isnan(self.propFac).sum()

    def propagate(self, F, z):
        #return ifftshift(ifftn(F*exp(self.propFac*z)))
        #print abs(F).sum()
        self._F[:] = fftshift(F*exp(self.propFac*z))
        self._plan_F_f()
        #print abs(self._f).sum()
        return ifftshift(self._f/sqrt(self._f.size))
        
    def propagate_r(self, f, z):
        #return ifftshift(ifftn(F*exp(self.propFac*z)))
        #figure()
        #imshow(angle(f))
        self._f[:] = fftshift(f)
        self._plan_f_F()
        #figure()
        #imshow(angle(self._F))
        return (ifftshift(self._F)*exp(-self.propFac*z))/sqrt(self._f.size)
        
FourierPropagator = FourierPropagatorHNA

def GenWidefieldAP(dx = 5):
    X, Y = meshgrid(arange(-2000, 2000., dx),arange(-2000, 2000., dx))
    u = 2*X/(dx*X.shape[0]*dx*pi)
    v = 2*Y/(dx*X.shape[1]*dx*pi)
    #print u.min()

    R = sqrt(u**2 + v**2)
    
    #print R.max()*lamb
    print (R/(n*lamb)).max()
    
    #imshow(R*lamb)
    #colorbar()
    
    k = 2*pi*n/lamb

    FP = FourierPropagator(u,v,k, lamb)

    #clf()
    #imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    M = 1.0*(R < (1.49/(n*lamb))) # NA/lambda
    
    #imshow(M)

    return X, Y, R, FP, M

def GenWidefieldPSF(zs, dx=5):
    X, Y, R, FP, F = GenWidefieldAP(dx)
    figure()
    imshow(abs(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    

def PsfFromPupil(pupil, zs, dx, lamb):
    dx = float(dx)
    X, Y = meshgrid(dx*arange(-pupil.shape[0]/2, pupil.shape[0]/2),dx*arange(-pupil.shape[1]/2, pupil.shape[1]/2))
    print X.min(), X.max()
    
    #print ps.shape
    #print arange(-ps.shape[0]/2, ps.shape[0]/2)
    u = 2*X/(dx*X.shape[0]*dx*pi)
    v = 2*Y/(dx*X.shape[1]*dx*pi)
    
    k = 2*pi*n/lamb

    #R = sqrt(u**2 + v**2)
    #M = 1.0*(R < (NA/(n*lamb))) # NA/lambda

    FP = FourierPropagator(u,v,k, lamb) 
    
    ps = concatenate([FP.propagate(pupil, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def PsfFromPupilVect(pupil, zs, dx, lamb, shape = [61,61]):
    dx = float(dx)
    X, Y = meshgrid(dx*arange(-pupil.shape[0]/2, pupil.shape[0]/2),dx*arange(-pupil.shape[1]/2, pupil.shape[1]/2))
    print X.min(), X.max()
    
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
    u = 2*X/(dx*X.shape[0]*dx*pi)
    v = 2*Y/(dx*X.shape[1]*dx*pi)

    R = sqrt(u**2 + v**2)
    
    phi = angle(u+ 1j*v)
    theta = arcsin(minimum(R*lamb, 1))
    
    #figure()
    #imshow(phi)
    
    #figure()
    #imshow(theta)
    
    ct = cos(theta)
    st = sin(theta)
    cp = cos(phi)
    sp = sin(phi)
    
    k = 2*pi*n/lamb
    
    
    #M = 1.0*(R < (NA/(n*lamb))) # NA/lambda

    FP = FourierPropagator(u,v,k, lamb) 
    
    fac = ct*cp**2 + sp**2
    ps = concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p = abs(ps**2)
    
    fac = (ct - 1)*cp*sp
    ps = concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = (ct - 1)*cp*sp
    ps = concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = ct*sp**2 + cp**2
    ps = concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = st*cp
    ps = concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)
    
    fac = st*sp
    ps = concatenate([FP.propagate(pupil*fac, z)[:,:,None] for z in zs], 2)
    p += abs(ps**2)

    return p[ox:ex, oy:ey, :] #abs(ps**2)

   
def ExtractPupil(ps, zs, dx, lamb=488, NA=1.3, n=1.51, nIters = 50, size=5e3):
    dx = float(dx)
    if not size:
        X, Y = meshgrid(float(dx)*arange(-ps.shape[0]/2, ps.shape[0]/2),float(dx)*arange(-ps.shape[1]/2, ps.shape[1]/2))
    else:
        X, Y = meshgrid(arange(-size, size, dx),arange(-size, size, dx))
    
    sx = ps.shape[0]
    sy = ps.shape[1]
    ox = (X.shape[0] - sx)/2
    oy = (X.shape[1] - sy)/2
    ex = ox + sx
    ey = oy + sy
    
    #print ps.shape
    #print arange(-ps.shape[0]/2, ps.shape[0]/2)
    u = 2*X/(dx*X.shape[0]*dx*pi)
    v = 2*Y/(dx*X.shape[1]*dx*pi)

    R = sqrt(u**2 + v**2)
    M = 1.0*(R < (NA/(n*lamb))) # NA/lambda
    
    k = 2*pi*n/lamb

    FP = FourierPropagator(u,v,k, lamb)

    pupil = M*exp(1j*0)
    
    sps = sqrt(ps)
    
    
    
    for i in range(nIters):
        new_pupil = 0*pupil
        
        bps = []
        abp = []
        
        print i#, abs(pupil).sum()
        
        #figure()
        #imshow(abs(pupil))
        res = 0
        
        for j in range(ps.shape[2]):
            #propogate to focal plane
            prop_j  = FP.propagate(pupil, zs[j])
            
            #print abs(prop_j).sum(), sqrt(ps[:,:,j]).sum()
            
            #print ps[:,:,j].shape
            #print prop_j.shape
            
            #figure()
            #imshow(angle(prop_j))
            
                        
            #print prop_j[ox:ex, oy:ey].shape, sps[:,:,j].shape
            
            res += ((abs(prop_j[ox:ex, oy:ey]) - sps[:,:,j])**2).sum()
            #replace amplitude, but keep phase
            prop_j[ox:ex, oy:ey] = sps[:,:,j]*exp(1j*angle(prop_j[ox:ex, oy:ey]))
            
            #print abs(prop_j).sum()
            
            #propagate back
            bp = FP.propagate_r(prop_j, zs[j])
            #figure()
            bps.append(abs(bp))
            abp.append(angle(bp))
            new_pupil += bp
            
            #figure()            
            #imshow(abs(new_pupil))
            
        new_pupil /= ps.shape[2]
        
        View3D(bps)
        View3D(abp)
        
        print 'res = %f' % (res/ps.shape[2])
        #print abs(new_pupil).sum()
        
        #np_A = abs(new_pupil)
        #np_A = ndimage.gaussian_filter(np_A,.5)
        #np_A *= M
        
        #np_P = angle(new_pupil)
        #np_P *= M
        #np_P = ndimage.gaussian_filter(np_P, .5)
        
        
        #imshow(abs(new_pupil))
        
        #crop to NA
        #new_pupil = new_pupil*M
        
        #pupil = np_A*exp(1j*np_P)
        pupil = new_pupil*M
    
    return pupil
    
    
    
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

def GenPRIPSF(zs, dx = 5, strength=1.0):
    X, Y, R, FP, F = GenWidefieldAP(dx)
    
    v = lamb*2*Y/(dx*Y.shape[0]*dx*pi)

    F = F * exp(-1j*sign(X)*10*strength*v)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenAstigPSF(zs, dx=5, strength=1.0):
    X, Y, R, FP, F = GenWidefieldAP(dx)
    
    u = lamb*2*X/(dx*X.shape[0]*dx*pi)    
    v = lamb*2*Y/(dx*Y.shape[0]*dx*pi)

    F = F * exp(-1j*((strength*v)**2 - 0.5*(strength*lamb*R)**2))
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenDHPSF(zs, dx=5, vortices=[0.0]):
    X, Y, R, FP, F = GenWidefieldAP(dx)
    
    u = lamb*2*X/(dx*X.shape[0]*dx*pi)    
    v = lamb*2*Y/(dx*Y.shape[0]*dx*pi)
    
    ph = 0*u
    
    for vc in vortices:
        ph += angle((u - vc) + 1j*v)

    F = F * exp(-1j*ph)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenCubicPhasePSF(zs, dx=5, strength=1.0):
    X, Y, R, FP, F = GenWidefieldAP(dx)
    
    v = lamb*2*Y/(dx*Y.shape[0]*dx*pi)
    u = lamb*2*X/(dx*X.shape[0]*dx*pi)

    F = F * exp(-1j*strength*(u**3 + v**3))
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenShiftedPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * exp(-1j*.01*Y)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenStripePRIPSF(zs, dx = 5):
    X, Y, R, FP, F = GenWidefieldAP(dx)

    F = F * exp(-1j*sign(sin(X))*.005*Y)
    #clf()
    #imshow(angle(F), cmap=cm.hsv)

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
   









