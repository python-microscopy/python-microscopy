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

NTHREADS = 1
FFTWFLAGS = ['measure']

n = 1.51
lamb = 680
k = 2*pi*n/lamb #k at 488nm

j = np.complex64(1j)

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
        #self.propFac = 1j*8*pi*sqrt(np.maximum((n/lamb)**2 - (u**2 + v**2), 0))
        #self.propFac = ((2*pi*n/lamb)*sqrt(np.maximum(1 - (u**2 + v**2), 0))).astype('f')
        self.propFac = ((2*pi*n/lamb)*cos(.5*pi*sqrt((u**2 + v**2)))).astype('f')
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
        #return ifftshift(ifftn(F*exp(self.propFac*z)))
        #print abs(F).sum()
        pf = self.propFac*float(z)
        fs = F*self.pfm*(cos(pf) + j*sin(pf))
        self._F[:] = fftshift(fs)
        #self._F[:] = (fs)
        self._plan_F_f()
        #print abs(self._f).sum()
        return ifftshift(self._f/sqrt(self._f.size))
        #return (self._f/sqrt(self._f.size))
        
    def propagate_r(self, f, z):
        #return ifftshift(ifftn(F*exp(self.propFac*z)))
        #figure()
        #imshow(angle(f))
        self._f[:] = fftshift(f)
        self._plan_f_F()
        #figure()
        #imshow(angle(self._F))
        pf = -self.propFac*float(z)
        return (ifftshift(self._F)*(cos(pf)+j*sin(pf)))/sqrt(self._f.size)
        
FourierPropagator = FourierPropagatorHNA

class FourierPropagatorClipHNA:
    def __init__(self, u,v,k, lamb = 488, n=1.51, field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0):
        #print k**2
        #m = (u**2 + v**2) <= (n/lamb**2)
        #self.propFac = fftw3f.create_aligned_array(u.shape, 'complex64')
        #self.propFac = 1j*8*pi*sqrt(np.maximum((n/lamb)**2 - (u**2 + v**2), 0))
        #self.propFac = ((2*pi*n/lamb)*sqrt(np.maximum(1 - (u**2 + v**2), 0))).astype('f')
        self.propFac = ((2*pi*n/lamb)*cos(.5*pi*sqrt((u**2 + v**2)))).astype('f')
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
        #return ifftshift(ifftn(F*exp(self.propFac*z)))
        #print abs(F).sum()
        pf = self.propFac*float(z)
        r = max(self.appR*(1 -self.apertureZGrad*z), 0)
        #print z, r
        M = (self.x*self.x + self.y*self.y) < (r*r)
        fs = F*M*self.pfm*(cos(pf) + j*sin(pf))
        self._F[:] = fftshift(fs)
        #self._F[:] = (fs)
        self._plan_F_f()
        #print abs(self._f).sum()
        return ifftshift(self._f/sqrt(self._f.size))
        #return (self._f/sqrt(self._f.size))
        
    def propagate_r(self, f, z):
        #return ifftshift(ifftn(F*exp(self.propFac*z)))
        #figure()
        #imshow(angle(f))
        self._f[:] = fftshift(f)
        self._plan_f_F()
        #figure()
        #imshow(angle(self._F))
        pf = -self.propFac*float(z)
        return (ifftshift(self._F)*(cos(pf)+j*sin(pf)))/sqrt(self._f.size)


def GenWidefieldAP(dx = 5, X=None, Y=None, lamb=700, n=1.51, NA = 1.47, apodization='sine'):
    if X == None or Y == None:
        X, Y = meshgrid(arange(-2000, 2000., dx),arange(-2000, 2000., dx))
    else:
        X, Y = meshgrid(X,Y)
    
    X = X - X.mean()
    Y = Y - Y.mean()
        
    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    #print u.min()

    R = sqrt(u**2 + v**2)
    
    #print R.max()*lamb
    #print(((R/(n*lamb)).max()))
    
    #imshow(R*lamb)
    #colorbar()
#    figure()
#    u_ = u[u.shape[0]/2, :]
#    plot(u_, u_)
#    plot(u_, sqrt(1 - u_**2))
#    plot(u_, sqrt(u_**2) < 1.49/2 )
#    plot(u_, sqrt(u_**2) < 1.49/n )
#    figure()
    
    k = 2*pi*n/lamb

    FP = FourierPropagator(u,v,k, lamb)

    #clf()
    #imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    if apodization == None:
        M = 1.0*(R < (NA/n)) # NA/lambda
    elif apodization == 'sine':
        M = 1.0*(R < (NA/n))*sqrt(cos(.5*pi*np.minimum(R, 1)))
    
    
    
    #M = M/M.sum()
    
    #imshow(M)

    return X, Y, R, FP, M, u, v
    
def GenWidefieldAPA(dx = 5, X=None, Y=None, lamb=700, n=1.51, NA = 1.47, field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0, apodizisation='sine'):
    if X == None or Y == None:
        X, Y = meshgrid(arange(-2000, 2000., dx),arange(-2000, 2000., dx))
    else:
        X, Y = meshgrid(X,Y)
    
    X = X - X.mean()
    Y = Y - Y.mean()
        
    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    #print u.min()

    R = sqrt(u**2 + v**2)
    
    #print R.max()*lamb
    #print(((R/(n*lamb)).max()))
    
    #imshow(R*lamb)
    #colorbar()
#    figure()
#    u_ = u[u.shape[0]/2, :]
#    plot(u_, u_)
#    plot(u_, sqrt(1 - u_**2))
#    plot(u_, sqrt(u_**2) < 1.49/2 )
#    plot(u_, sqrt(u_**2) < 1.49/n )
#    figure()
    
    k = 2*pi*n/lamb

    FP = FourierPropagatorClipHNA(u,v,k, lamb, n, field_x, field_y, apertureNA, apertureZGradient)

    #clf()
    #imshow(imag(FP.propFac))
    #colorbar()

    #apperture mask
    M = 1.0*(R < (NA/n)) # NA/lambda
    
    if apodizisation == None:
        M = 1.0*(R < (NA/n)) # NA/lambda
    elif apodizisation == 'sine':
        M = 1.0*(R < (NA/n))*sqrt(cos(.5*pi*np.minimum(R, 1)))
    
    #M = M/M.sum()
    
    #imshow(M)

    return X, Y, R, FP, M, u, v

def GenWidefieldPSF(zs, dx=5, lamb=700, n=1.51, NA = 1.47,apodization=None):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n=n, NA = NA, apodization=apodization)
    #figure()
    #imshow(abs(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenWidefieldPSFA(zs, dx=5, lamb=700, n=1.51, NA = 1.47,field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0):
    X, Y, R, FP, F, u, v = GenWidefieldAPA(dx, lamb=lamb, n=n, NA = NA, field_x=field_x, field_y=field_y, apertureNA=apertureNA, apertureZGradient = apertureZGradient)
    #figure()
    #imshow(abs(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    

def PsfFromPupil(pupil, zs, dx, lamb):
    dx = float(dx)
    X, Y = meshgrid(dx*arange(-pupil.shape[0]/2, pupil.shape[0]/2),dx*arange(-pupil.shape[1]/2, pupil.shape[1]/2))
    print((X.min(), X.max()))
    
    X = X - X.mean()
    Y = Y - Y.mean()
    
    #print ps.shape
    #print arange(-ps.shape[0]/2, ps.shape[0]/2)

    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)
    
    k = 2*pi*n/lamb

    #R = sqrt(u**2 + v**2)
    #M = 1.0*(R < (NA/(n*lamb))) # NA/lambda

    FP = FourierPropagator(u,v,k, lamb) 
    
    ps = concatenate([FP.propagate(pupil, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def PsfFromPupilVect(pupil, zs, dx, lamb, shape = [61,61]):
    dx = float(dx)
    X, Y = meshgrid(dx*arange(-pupil.shape[0]/2, pupil.shape[0]/2),dx*arange(-pupil.shape[1]/2, pupil.shape[1]/2))
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
    
def PsfFromPupilVectFP(X, Y, R, FP, u, v, n, pupil, zs):
    
    phi = angle(u+ 1j*v)
    theta = arcsin(minimum(R/n, 1))
    
    #figure()
    #imshow(phi)
    
    figure()
    #imshow(theta)
    print theta.min(), theta.max()
    
    ct = cos(theta)
    st = sin(theta)
    cp = cos(phi)
    sp = sin(phi) 
    
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

    return p#p[ox:ex, oy:ey, :] #abs(ps**2)

def PsfFromPupilFP(X, Y, R, FP, u, v, n, pupil, zs):

    ps = concatenate([FP.propagate(pupil, z)[:,:,None] for z in zs], 2)
    p = abs(ps**2)

    return p#p[ox:ex, oy:ey, :] #abs(ps**2)
   
def ExtractPupil(ps, zs, dx, lamb=488, NA=1.3, n=1.51, nIters = 50, size=5e3):
    dx = float(dx)
    if not size:
        X, Y = meshgrid(float(dx)*arange(-ps.shape[0]/2, ps.shape[0]/2),float(dx)*arange(-ps.shape[1]/2, ps.shape[1]/2))
    else:
        X, Y = meshgrid(arange(-size, size, dx),arange(-size, size, dx))
        
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
    
    

    R = sqrt(u**2 + v**2)
    M = 1.0*(R < (NA/n)) # NA/lambda
    
    figure()
    imshow(R)
    
    colorbar()
    contour(M, [0.5])
    figure()
    
    k = 2*pi*n/lamb

    FP = FourierPropagator(u,v,k, lamb)

    pupil = M*exp(1j*0)
    
    sps = sqrt(ps)
    
    View3D(sps)
    
    for i in range(nIters):
        new_pupil = 0*pupil
        
        bps = []
        abp = []
        
        print(i)#, abs(pupil).sum()
        
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
        
        #View3D(bps)
        #View3D(abp)
        
        print(('res = %f' % (res/ps.shape[2])))
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
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)
    
    theta = angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
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
    
def GenZernikeDPSF(zs, dx = 5, zernikeCoeffs = {}, lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    from PYME.misc import zernike, snells
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)
    
    theta = angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    if ns == n:
        T = 1.0*F
    else:
        #find angles    
        t_t = np.minimum(r*arcsin(NA/n), np.pi/2)
        
        #corresponding angle in sample with mismatch
        t_i = snells.theta_t(t_t, n, ns)
        
        #Transmission at interface (average of S and P)
        T = 0.5*(snells.Ts(t_i, ns, n) + snells.Tp(t_i, ns, n))
        #concentration of high angle rays:
        T = T*F/(n*np.cos(t_t)/np.sqrt(ns*2 - (n*np.sin(t_t))**2))
    
    
    
    #imshow(T*(-t_i + snells.theta_t(t_t+.001, n, ns)))
    #imshow(F)
    #colorbar()
    #figure()
    #imshow(T, clim=(.8, 1.2))
    #colorbar()
    #figure()
    #imshow(T*(-t_i + snells.theta_t(t_t+.01, n, ns)))
    #imshow(t_i - t_t)
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c*zernike.zernike(i, r, theta)
        
    #clf()
    #imshow(angle(exp(1j*ang)))
        
    F = T*exp(-1j*ang)
        
    #figure()
    #imshow(angle(F))
        
    if vect:
        return PsfFromPupilVectFP(X,Y,R, FP, u,v, n, F, zs)
    else:
        return PsfFromPupilFP(X,Y,R, FP, u,v, n, F, zs)

#    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)
#    
#    if beadsize == 0:
#        return abs(ps**2)
#    else:
#        p1 = abs(ps**2)
        
        
    
def GenZernikeDAPSF(zs, dx = 5, X=None, Y=None, zernikeCoeffs = {}, lamb=700, n=1.51, NA = 1.47, ns=1.51,field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0, apodizisation=None, vect=True):
    from PYME.misc import zernike, snells
    X, Y, R, FP, F, u, v = GenWidefieldAPA(dx, X = X, Y=Y, lamb=lamb, n = n, NA = NA, field_x=field_x, field_y=field_y, apertureNA=apertureNA, apertureZGradient = apertureZGradient, apodizisation=apodizisation)
    
    theta = angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    if ns == n:
        T = 1.0*F
    else:
        #find angles    
        t_t = np.minimum(r*arcsin(NA/n), np.pi/2)
        
        #corresponding angle in sample with mismatch
        t_i = snells.theta_t(t_t, n, ns)
        
        #Transmission at interface (average of S and P)
        T = 0.5*(snells.Ts(t_i, ns, n) + snells.Tp(t_i, ns, n))
        #concentration of high angle rays:
        T = T*F/(n*np.cos(t_t)/np.sqrt(ns*2 - (n*np.sin(t_t))**2))
    
    
    
    #imshow(T*(-t_i + snells.theta_t(t_t+.001, n, ns)))
    #imshow(F)
    #colorbar()
    #figure()
    #imshow(T, clim=(.8, 1.2))
    #colorbar()
    #figure()
    #imshow(T*(-t_i + snells.theta_t(t_t+.01, n, ns)))
    #imshow(t_i - t_t)
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c*zernike.zernike(i, r, theta)
        
    #clf()
    #imshow(angle(exp(1j*ang)))
        
    F = T*exp(-1j*ang)
        
    #figure()
    #imshow(angle(F))
        
    if vect:
        return PsfFromPupilVectFP(X,Y,R, FP, u,v, n, F, zs)
    else:
        return PsfFromPupilFP(X,Y,R, FP, u,v, n, F, zs)

    #ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    #return abs(ps**2)

def GenPRIPSF(zs, dx = 5, strength=1.0, dp=0, lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    #X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA=NA)
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)

    F = F * exp(-1j*sign(X)*(10*strength*v + dp/2))
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenICPRIPSF(zs, dx = 5, strength=1.0, NA=1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA = NA)

    F = F * exp(-1j*sign(X)*10*strength*v)
    #clf()
    #imshow(angle(F))
    
    F_ = F
    F = F*(X >= 0)

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    p1 = abs(ps**2)
    
    F = F_*(X < 0)

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return  p1 + abs(ps**2)
    
def GenColourPRIPSF(zs, dx = 5, strength=1.0, transmit = [1,1]):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)

    F = F * exp(-1j*sign(X)*10*strength*v)
    
    F = F*(sqrt(transmit[0])*(X < 0) +  sqrt(transmit[1])*(X >=0))
    #clf()
    imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenAstigPSF(zs, dx=5, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)

    F = F * exp(-1j*((strength*v)**2 - 0.5*(strength*R)**2))
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenSAPSF(zs, dx=5, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    theta = angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)

    F = F * exp(-1j*strength*z8)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenR3PSF(zs, dx=5, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    theta = angle(X + 1j*Y)
    
    z8 = r**3#zernike.zernike(8, r, theta)

    F = F * exp(-1j*strength*z8)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenBesselPSF(zs, dx=5, rad=.95, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    #from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    #theta = angle(X + 1j*Y)
    
    #z8 = zernike.zernike(8, r, theta)

    F = F * (r > rad)#exp(-1j*strength*z8)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenSABesselPSF(zs, dx=5, rad=.95, strength=1.0, X=None, Y=None, lamb=700, n=1.51, NA = 1.47):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, lamb=lamb, n=n, NA = NA)
    
    r = R/R[abs(F)>0].max()
    theta = angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)

    F = F * (r > rad)*exp(-1j*strength*z8)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

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
        theta = angle(X + 1j*Y)
        
        z8 = zernike.zernike(8, r, theta)
        a_s = (v**2 - 0.5*R**2)
        
        fps[Xk] = (fpset, z8, a_s)
    else:
        fpset, z8, a_s = fps[Xk]
        X, Y, R, FP, F, u, v = fpset #GenWidefieldAP(dx, X, Y)
    

    #F = F * exp(-1j*(strength*a_s + SA*z8))
    pf = -(strength*a_s + SA*z8)
    F = F *(cos(pf) + j*sin(pf))
    
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

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

    F = F * exp(-1j*sign(X)*10*strength*v)
    
    theta = angle(X + 1j*Y)
    
                
    ang = SA*zernike.zernike(8, r, theta)
            
    F = F*exp(-1j*ang)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenDHPSF(zs, dx=5, vortices=[0.0], lamb=700, n=1.51, NA = 1.47, ns=1.51, beadsize=0, vect=False, apodization=None):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, lamb=lamb, n = n, NA = NA, apodization=apodization)
    
    ph = 0*u
    
    for vc in vortices:
        ph += angle((u - vc) + 1j*v)

    F = F * exp(-1j*ph)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenCubicPhasePSF(zs, dx=5, strength=1.0, X = None, Y = None, n=1.51, NA = 1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, X, Y, n = n, NA = NA)

    F = F * exp(-1j*strength*(u**3 + v**3))
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)

def GenShiftedPSF(zs, dx = 5):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)

    F = F * exp(-1j*.01*Y)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
    
def GenBiplanePSF(zs, dx = 5, zshift = 500, xshift = 1, NA=1.47):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx, NA=NA)
    F_ = F

    F = F * exp(-1j*.01*xshift*Y)
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z-zshift/2)[:,:,None] for z in zs], 2)

    ps1 = abs(ps**2)
    
    F = F_ * exp(1j*.01*xshift*Y) 
    
    #clf()
    #imshow(angle(F))

    ps = concatenate([FP.propagate(F, z+zshift/2)[:,:,None] for z in zs], 2)
    return 0.5*ps1 +0.5*abs(ps**2)

def GenStripePRIPSF(zs, dx = 5):
    X, Y, R, FP, F, u, v = GenWidefieldAP(dx)

    F = F * exp(-1j*sign(sin(X))*.005*Y)
    #clf()
    #imshow(angle(F), cmap=cm.hsv)

    ps = concatenate([FP.propagate(F, z)[:,:,None] for z in zs], 2)

    return abs(ps**2)
   









