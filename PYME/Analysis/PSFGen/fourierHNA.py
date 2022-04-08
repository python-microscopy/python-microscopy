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
#from pylab import *
#import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import ifftshift, ifftn, fftn, fftshift
import warnings

import PYME.misc.fftw_compat as fftw3f
from PYME.Deconv import fftwWisdom

fftwWisdom.load_wisdom()

NTHREADS = 1
FFTWFLAGS = ['measure']

j = np.complex64(1j)

# class FourierPropagator:
#     def __init__(self, u,v,k):
#          self.propFac = -1j*(2*2**2*(u**2 + v**2)/k)
#
#     def propagate(self, F, z):
#         return ifftshift(ifftn(F*np.exp(self.propFac*z)))
        
class FourierPropagatorHNA:
    _propagator_cache = {} #NB must re-define this in derived classes so that they use a separate cache
    
    def __init__(self, u,v,k=None, lamb = 488, n=1.51):
        """A FourierPropagator object allows us evaluate the electric field at a given defocus by propagating a complex
        pupil distribution a given distance from the nominal focus by adding the relevant phase term to the pupil and
        then taking the Fourier amplitude.
        
        Parameters
        ==========
        
        u, v : 2d arrays of float
            the co-ordinates in spatial frequencies within the pupil plane
            
        lamb : float
            the wavelength in nm
            
        n : float
            the refractive index of the media
            
            
        Notes
        =====
        
        u, v must be the same size as the eventual pupil distribution to be used. On creation, the FourierPropagator
        pre-calculates the phase factor to add for each u, v, co-ordinate and also pre-computes FFTW3 plans for the
        necessary Fourier transforms.
        
        """
        if not k is None:
            raise DeprecationWarning('k is no longer used')
        
        
        #R = np.sqrt(u**2 + v**2)
        self.propFac = ((2 * np.pi * n / lamb) * np.sqrt(1 - np.minimum(u ** 2 + v ** 2, 1))).astype('f')
        
        #self.pfm =(self.propFac > 0).astype('f')
        self.pfm = (np.sqrt(u ** 2 + v ** 2) < 1).astype('f')

        self._F = fftw3f.create_aligned_array(u.shape, 'complex64')
        self._f = fftw3f.create_aligned_array(u.shape, 'complex64')
        
        #print('Creating plans for FFTs - this might take a while')

        #calculate plans for other ffts
        self._plan_f_F = fftw3f.Plan(self._f, self._F, 'forward', flags = FFTWFLAGS, nthreads=NTHREADS)
        self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        #self._plan_F_f = fftw3f.Plan(self._F, self._f, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()

    def propagate(self, F, z):
        """ Propagate a complex pupil, F,  a distance z from the nominal focus and return the electric field amplitude
        
        Parameters
        ==========
        
        F : 2D array
            complex pupil
            
        z : float
            distance in nm to propagate
        
        """
        pf = self.propFac*float(z)
        fs = F*self.pfm*(np.cos(pf) + j*np.sin(pf))
        self._F[:] = ifftshift(fs)

        self._plan_F_f()
        return fftshift(self._f/np.sqrt(self._f.size))
        
    def propagate_r(self, f, z):
        """
        Backpropagate an electric field distribution, f, at defocus z to the nominal focus and return the complex pupil
        
        Parameters
        ----------
        f : 2D array
            complex electric field amplitude
        z : float
            nominal distance of plane from focus in nm

        Returns
        -------

        """
        self._f[:] = fftshift(f)
        self._plan_f_F()
        
        pf = -self.propFac*float(z)
        return (ifftshift(self._F)*(np.cos(pf)+j*np.sin(pf)))/np.sqrt(self._f.size)
    
    @classmethod
    def get_propagator(cls, u,v,k=None, lamb = 488, n=1.51):
        cache_key = (u.shape, u[0,0], lamb, n)
        
        try:
            return cls._propagator_cache[cache_key]
        except KeyError:
            cls._propagator_cache[cache_key] = cls(u,v,k=k, lamb = lamb, n=n)
            return cls._propagator_cache[cache_key]
        
FourierPropagator = FourierPropagatorHNA

class FourierPropagatorClipHNA(FourierPropagatorHNA):
    _propagator_cache = {}
    def __init__(self, u,v,k=None, lamb = 488, n=1.51, field_x=0, field_y=0, apertureNA=1.5, apertureZGradient = 0):
        """A FourierPropagator object allows us evaluate the electric field at a given defocus by propagating a complex
        pupil distribution a given distance from the nominal focus by adding the relevant phase term to the pupil and
        then taking the Fourier amplitude.
        
        This version will also clip the propagated pupil at a specified NA, potentially off-centered. This is useful in
        simulating clipping / vignetting effects within the objective.
        
        Parameters
        ----------
        u, v : 2D arrays
            co-ordinates in the pupil plane
        lamb : float
            wavelength in nm
        n : float
            refractive index
        field_x, field_y : float
            how much our aperture is off-center, in pupil coordinates
        apertureNA : float
            the NA of our clipping aperture
        apertureZGradient : float
            a factor by which the NA of the clipping aperture changes with defocus. Permits modelling of apertures which
            are not in the pupil plane, as these will look like an aperture which changes with defocus.
        """
        if not k is None:
            raise DeprecationWarning('k is no longer used')

        
        self.appR = apertureNA/n
        self.apertureZGrad = apertureZGradient
        self.x = u - field_x
        self.y = v - field_y
        
        FourierPropagatorHNA.__init__(self, u, v, lamb=lamb, n=n)


    def propagate(self, F, z):
        """
        Propagate a complex pupil, F,  a distance z from the nominal focus and return the electric field amplitude

        Parameters
        ==========

        F : 2D array
            complex pupil

        z : float
            distance in nm to propagate

        """
        pf = self.propFac*float(z)
        r = max(self.appR*(1 -self.apertureZGrad*z), 0)
        
        M = (self.x*self.x + self.y*self.y) < (r*r)
        fs = F*M*self.pfm*(np.cos(pf) + j*np.sin(pf))
        self._F[:] = fftshift(fs)

        self._plan_F_f()
        
        return ifftshift(self._f/np.sqrt(self._f.size))
        
    
##############
####diffhelper

def _apodization_function(R, NA, n, apodization='sine', ns=None):
    from PYME.misc import snells
    t_ = np.arcsin(np.minimum(R, 1))
    
    if (apodization is None) or (apodization == 'none'):
        M = 1.0 * (R < (NA / n)) # NA/lambda
    elif apodization == 'sine':
        M = 1.0 * (R < (NA / n)) * np.sqrt(np.cos(t_))
    elif apodization == 'empirical':
        r_ = np.minimum(R, 1)
        M = 1.0 * (R < (NA / n)) * (1 - 0.65 * t_) * (1 - np.exp(-10 * ((NA / n) - r_)))
    else:
        M = 1.0 * (R < (NA / n)) # NA/lambda

    r = R / R[abs(M) > 0].max()

    if (ns is None) or (ns == n):
        T = 1.0 * M
    else:
        #find angles
        t_t = np.minimum(r * np.arcsin(NA / n), np.pi / 2)
    
        #corresponding angle in sample with mismatch
        t_i = snells.theta_t(t_t, n, ns)
    
        #Transmission at interface (average of S and P)
        T = 0.5 * (snells.Ts(t_i, ns, n) + snells.Tp(t_i, ns, n))
        #concentration of high np.angle rays:
        T = T * 1.0 / (n * np.cos(t_t) / np.sqrt(ns * 2 - (n * np.sin(t_t)) ** 2))
    
    return M*T
    


def widefield_pupil_and_propagator(dx = 5, X=None, Y=None, lamb=700, n=1.51, NA=1.47, **kwargs):
    """
    Generate a widefield pupil and corresponding propagator
    
    Parameters
    ----------
    dx : float
        pixel size in resulting image
    X, Y : 1D arrays (optional)
        pixel coordinates in image plane  nm. Note that if provided, the pixel size should match that in dx. If not
        provided, co-ordinated will be generated for a default size of 4umx4um
    lamb : float
        wavelength
    n : float
        refractive index
    NA : float
        Numerical aperture

    Returns
    -------
    X, Y : pixel coordinates in nm
    R : radial co-ordinates in u,v space
    FP : the propagator
    M : the widefield pupil
    u, v : co-ordinates in pupil space
    """
    
    if X is None or Y is None:
        xs, ys, zs = kwargs.get('output_shape', (61, 61, 61))
        
        X, Y = np.meshgrid(float(dx)*(np.arange(xs) - np.floor(xs/2)),float(dx)*(np.arange(ys) - np.floor(ys/2)))
    else:
        dx = X[1] - X[0]
        X, Y = np.meshgrid(X,Y)
    
    X = X - X.mean()
    Y = Y - Y.mean()

    u = X*lamb/(n*X.shape[0]*dx*dx)
    v = Y*lamb/(n*X.shape[1]*dx*dx)

    R = np.sqrt(u**2 + v**2)

    FP = FourierPropagator.get_propagator(u, v, lamb=lamb, n=n)

    M = 1.0 * (R < (NA / n))
    
    return X, Y, R, FP, M, u, v
    
def clipped_widefield_pupil_and_propagator(dx = 5, X=None, Y=None, lamb=700, n=1.51, NA = 1.47, field_x=0, field_y=0,
                                           apertureNA=1.5, apertureZGradient = 0, **kwargs):
    """
        Generate a widefield pupil and corresponding propagator for the case where the light is clipped with another
        aperture which is off-centre and not in the pupil plane

        Parameters
        ----------
        dx : float
            pixel size in resulting image
        X, Y : 1D arrays (optional)
            pixel coordinates in image plane  nm. Note that if provided, the pixel size should match that in dx. If not
            provided, co-ordinated will be generated for a default size of 4umx4um
        lamb : float
            wavelength
        n : float
            refractive index
        NA : float
            Numerical aperture
        apodization : string or None
            one of None or 'sine'. Pupil apodization.

        Returns
        -------
        X, Y : pixel coordinates in nm
        R : radial co-ordinates in u,v space
        FP : the propagator
        M : the widefield pupil
        u, v : co-ordinates in pupil space
        """

    X, Y, R, FP, M, u, v = widefield_pupil_and_propagator(dx = dx, X=X, Y=Y, lamb=lamb, n=n, NA=NA)

    FP = FourierPropagatorClipHNA.get_propagator(u,v, None, lamb, n, field_x, field_y, apertureNA, apertureZGradient)

    return X, Y, R, FP, M, u, v


##############################
# Gerchberg-Saxton pupil extraction

def ExtractPupil(ps, zs, dx, lamb=488, NA=1.3, n=1.51, nIters=50, size=5e3, intermediateUpdates=False, apodization='sine'):
    dx = float(dx)
    
    if not size:
        X, Y = np.meshgrid(float(dx) * np.arange(-ps.shape[0] / 2, ps.shape[0] / 2),
                           float(dx) * np.arange(-ps.shape[1] / 2, ps.shape[1] / 2))
    else:
        X, Y = np.meshgrid(np.arange(-size, size, dx), np.arange(-size, size, dx))
    
    X = X - X.mean()
    Y = Y - Y.mean()
    
    sx = ps.shape[0]
    sy = ps.shape[1]
    ox = int((X.shape[0] - sx) / 2)
    oy = int((X.shape[1] - sy) / 2)
    ex = ox + sx
    ey = oy + sy
    
    u = X * lamb / (n * X.shape[0] * dx * dx)
    v = Y * lamb / (n * X.shape[1] * dx * dx)
    
    R = np.sqrt(u ** 2 + v ** 2)
    M = 1.0 * (R < (NA / n)) # NA/lambda
    
    FP = FourierPropagator(u, v, lamb=lamb, n=n)
    
    P = M*_apodization_function(R, NA, n, apodization)
    pupil = P * np.exp(1j * 0)
    
    sps = np.sqrt(ps)
    
    #normalize
    sps = np.pi * M.sum() * sps / sps.sum(1).sum(0)[None, None, :]
    
    for i in range(nIters):
        new_pupil = 0 * pupil
        
        print(i)#, abs(pupil).sum()
        
        res = 0
        
        jr = np.argsort(np.random.rand(ps.shape[2]))
        
        for j in jr:#range(ps.shape[2]):
            #propogate to focal plane
            prop_j = FP.propagate(pupil, zs[j])
            
            pj = prop_j[ox:ex, oy:ey]
            pj_mag = abs(pj)
            sps_j = sps[:, :, j]
            
            res += ((pj_mag - sps_j) ** 2).sum()
            
            #replace amplitude, but keep phase
            prop_j[ox:ex, oy:ey] = sps_j * np.exp(1j * np.angle(pj))
            
            #propagate back
            bp = FP.propagate_r(prop_j, zs[j])
            
            new_pupil += bp
            
            if intermediateUpdates:
                pupil = P* np.exp(1j * M * np.angle(bp))
        
        new_pupil /= ps.shape[2]
        
        print(('res = %f' % (res / ps.shape[2])))
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
        pupil = P * np.exp(1j * M * np.angle(new_pupil))
    
    return pupil

##################
# Generating PSFs from pupils

def _gen_bead_pupil(X, Y, beadsize):
    r2 = X*X + Y*Y

    dx = X[1, 0] - X[0, 0]
    
    bead_proj = np.sqrt(np.maximum(beadsize*beadsize - (r2 - dx*dx), 0))

    return abs(fftshift(fftn(bead_proj)))

_bead_cache = {}

def get_bead_pupil(X, Y, beadsize):
    dx = X[1, 0] - X[0, 0]
    bp = _bead_cache.get((X.shape, dx, beadsize), None)
    
    if bp is None:
        bp = _gen_bead_pupil(X, Y, beadsize)
        _bead_cache[(X.shape, dx, beadsize)] = bp
    
    return bp


def _crop_to_shape(p, output_shape):
    #crop to output shape
    if output_shape is None:
        return p
    else:
        sx = output_shape[0]
        sy = output_shape[1]
        ox = int(np.ceil((p.shape[0] - sx) / 2.0))
        oy = int(np.ceil((p.shape[1] - sy) / 2.0))
        ex = ox + sx
        ey = oy + sy
        
        return p[ox:ex, oy:ey, :]

def _PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil, zs, n=1.51, NA=1.47, vectorial=False, apodization='sine', ns = None, output_shape=None, beadsize=0,**kwargs):
    pupil = pupil*_apodization_function(R, NA, n, apodization, ns=ns)
    
    if beadsize > 0:
        pupil = pupil*get_bead_pupil(X, Y, beadsize)
        
    if vectorial:
        if vectorial is True:
            #use circular pol by default
            a = 1
            b = -1j
        else:
            #vectorial is a tuple of polarization components
            a, b = vectorial
        
        phi = np.angle(u + 1j * v)
        theta = np.arcsin(np.minimum(R, 1))
        
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        
        #ax
        fac = a*(ct * cp ** 2 + sp ** 2)
        ps_x = np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p = abs(ps ** 2)
        
        #ay
        fac = a*(ct - 1) * cp * sp
        ps_y = np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p += abs(ps ** 2)
        
        #az
        fac = -a*st * cp
        ps_z = np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p += abs(ps ** 2)
        
        
        #bx
        fac = b*(ct - 1) * cp * sp
        ps_x += np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        
        p = abs(ps_x ** 2)
        
        #by
        fac = b*(ct * sp ** 2 + cp ** 2)
        ps_y += np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        p += abs(ps_y ** 2)
        
        
        #bz
        fac = -b*st * sp
        ps_z += np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        p += abs(ps_z ** 2)
        
        
    else:
        ###########
        # Default scalar case
        ps = np.concatenate([FP.propagate(pupil, z)[:, :, None] for z in zs], 2)
        p = abs(ps ** 2)

    return _crop_to_shape(p, output_shape)




def E_field_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil, zs, n=1.51, NA=1.47, vectorial=False, apodization='sine',
                                  ns=None, output_shape=None, beadsize=0, **kwargs):
    pupil = pupil * _apodization_function(R, NA, n, apodization, ns=ns)
    
    if beadsize > 0:
        pupil = pupil * get_bead_pupil(X, Y, beadsize)
    
    if vectorial:
        if vectorial is True:
            #use circular pol by default
            a = 1
            b = -1j
        else:
            #vectorial is a tuple of polarization components
            a, b = vectorial
        
        phi = np.angle(u + 1j * v)
        theta = np.arcsin(np.minimum(R, 1))
        
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        
        #ax
        fac = a * (ct * cp ** 2 + sp ** 2)
        ps_x = np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p = abs(ps ** 2)
        
        #ay
        fac = a * (ct - 1) * cp * sp
        ps_y = np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p += abs(ps ** 2)
        
        #az
        fac = -a * st * cp
        ps_z = np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p += abs(ps ** 2)
        
        
        #bx
        fac = b * (ct - 1) * cp * sp
        ps_x += np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        
        #p = abs(ps_x ** 2)
        
        #by
        fac = b * (ct * sp ** 2 + cp ** 2)
        ps_y += np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p += abs(ps_y ** 2)
        
        #bz
        fac = -b * st * sp
        ps_z += np.concatenate([FP.propagate(pupil * fac, z)[:, :, None] for z in zs], 2)
        #p += abs(ps_z ** 2)
        
        return _crop_to_shape(ps_x, output_shape), _crop_to_shape(ps_y, output_shape), _crop_to_shape(ps_z, output_shape)
    
    
    else:
        ###########
        # Default scalar case
        ps = np.concatenate([FP.propagate(pupil, z)[:, :, None] for z in zs], 2)
        return _crop_to_shape(ps, output_shape)


def EField_from_pupil_and_propagator(X, Y, R, FP, u, v, n, NA, pupil, zs, apodization='sine', ):
    pupil = pupil * _apodization_function(R, NA, n, apodization)
    
    ps = np.concatenate([FP.propagate(pupil, z)[:, :, None] for z in zs], 2)
    
    return ps

def _psf_from_e_field(ef):
    if len(ef) == 3: #vectorial
        e_x, e_y, e_z = ef
    
        return np.abs(e_x ** 2) + np.abs(e_y ** 2) + np.abs(e_z ** 2)
    else:
        return np.abs(ef**2)


def PSF_from_pupil_and_propagator(*args, **kwargs):
    e_f = E_field_from_pupil_and_propagator(*args, **kwargs)
    return _psf_from_e_field(e_f)

def PsfFromPupil(pupil, zs, dx, **kwargs):
    dx = float(dx)
    
    n = kwargs.get('n', 1.51)
    lamb = kwargs.get('lamb', 700.)
    
    X, Y = np.meshgrid(dx * np.arange(-pupil.shape[0] / 2, pupil.shape[0] / 2),
                       dx * np.arange(-pupil.shape[1] / 2, pupil.shape[1] / 2))
    print((X.min(), X.max()))
    
    X = X - X.mean()
    Y = Y - Y.mean()
    
    u = X * lamb / (n * X.shape[0] * dx * dx)
    v = Y * lamb / (n * X.shape[1] * dx * dx)
    
    R = np.sqrt(u ** 2 + v ** 2)
    
    # default to no apodization when generating a PSF from a given (usually extracted) pupil on the basis
    # that the apodization will have already been extracted in the phase retrieval process
    # CHECKME - does our phase retrieval still do this?
    kwargs['apodization'] = kwargs.get('apodization', None)
    
    FP = FourierPropagator.get_propagator(u, v, lamb=lamb, n=n)
    
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=pupil, zs=zs, **kwargs)
     

def PsfFromPupilVect(pupil, zs, dx, shape=[61, 61], **kwargs):
    warnings.warn('PsfFromPupilVect is deprecated, use PsfFromPupil(...,vectorial=True) instead',DeprecationWarning)
    return PsfFromPupil(pupil, zs, dx,output_shape=shape, vectorial=True, **kwargs)

##########################
###PSF Generation functions

def GenWidefieldPSF(zs, **kwargs):
    X, Y, R, FP, pupil, u, v = widefield_pupil_and_propagator(**kwargs)
    
    kwargs.pop('X', None)
    kwargs.pop('Y', None)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=pupil, zs=zs, **kwargs)

def Gen4PiPSF(zs,phi=0, zernikeCoeffs=[{},{}], **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, pupil, u, v = widefield_pupil_and_propagator(**kwargs)
    
    dphi = phi

    kwargs.pop('X', None)
    kwargs.pop('Y', None)
    
    NA = kwargs.get('NA', 1.47)
    n = kwargs.get('n', 1.51)
    output_shape = kwargs.get('output_shape', None)
    beadsize = kwargs.get('beadsize', 0)
    vectorial = kwargs.get('vectorial', False)

    pupil = pupil * _apodization_function(R, NA, n, kwargs.get('apodization', None))
    
    if beadsize > 0:
        pupil = pupil * get_bead_pupil(X, Y, beadsize)

    theta = np.angle(X + 1j * Y)
    r = R / R[abs(pupil) > 0].max()



    ang_u = 0
    ang_l = 0
    
    zerns_upper, zerns_lower = zernikeCoeffs
    
    #print zernikeCoeffs, zerns_upper, zerns_lower

    for i, c in zerns_upper.items():
        ang_u = ang_u + c * zernike.zernike(i, r, theta)
        
    for i, c in zerns_lower.items():
        ang_l = ang_l + c * zernike.zernike(i, r, theta)
        
    pupil_upper = pupil * np.exp(-1j * ang_u)
    pupil_lower = pupil * np.exp(-1j * ang_l)
        
    #pupil_upper = pupil
    #pupil_lower = pupil

    if vectorial:
        if vectorial is True:
            #use circular pol by default
            a = 1
            b = -1j
        else:
            #vectorial is a tuple of polarization components
            a, b = vectorial
        
        phi = np.angle(u + 1j * v)
        theta = np.arcsin(np.minimum(R, 1))
    
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        
        psf = []
        for z in zs:
            #ax
            fac = a*(ct * cp ** 2 + sp ** 2)
            ps_u = FP.propagate(pupil_upper * fac, z)
            ps_l = FP.propagate(pupil_lower * fac, -z)*np.exp(1j*dphi)
            ps_x = ps_u + ps_l
            #p = abs(ps ** 2)
        
            #ay
            fac = a*(ct - 1) * cp * sp
            ps_u = FP.propagate(pupil_upper * fac, z)
            ps_l = FP.propagate(pupil_lower * fac, -z) * np.exp(1j * dphi)
            ps_y = ps_u + ps_l
            #p += abs(ps ** 2)
        
            #bx
            fac = b*(ct - 1) * cp * sp
            ps_u = FP.propagate(pupil_upper * fac, z)
            ps_l = FP.propagate(pupil_lower * fac, -z) * np.exp(1j * dphi)
            ps_x += (ps_u + ps_l)
            p = abs(ps_x ** 2)
        
            #by
            fac = b*(ct * sp ** 2 + cp ** 2)
            ps_u = FP.propagate(pupil_upper * fac, z)
            ps_l = FP.propagate(pupil_lower * fac, -z) * np.exp(1j * dphi)
            ps_y += (ps_u + ps_l)
            p += abs(ps_y ** 2)
        
            #az
            fac = -a*st * cp
            ps_u = FP.propagate(pupil_upper * fac, z)
            ps_l = FP.propagate(pupil_lower * fac, -z) * np.exp(1j * dphi)
            ps_z = ps_u + ps_l
            #p += abs(ps ** 2)
        
            fac = -b*(st * sp)
            ps_u = FP.propagate(pupil_upper * fac, z)
            ps_l = FP.propagate(pupil_lower * fac, -z) * np.exp(1j * dphi)
            ps_z += (ps_u + ps_l)
            p += abs(ps_z ** 2)
            
            psf.append(p[:,:,None])

        p = np.concatenate(psf, 2)
    else:
        ###########
        # Default scalar case
        ps_u = np.concatenate([FP.propagate(pupil_upper, z)[:, :, None] for z in zs], 2)
        ps_l = np.concatenate([FP.propagate(pupil_lower, -z)[:, :, None] * np.exp(1j * dphi) for z in zs], 2)
        ps = ps_u + ps_l
        p = abs(ps ** 2)
    
    if output_shape is None:
        return p
    else:
        sx = output_shape[0]
        sy = output_shape[1]
        ox = np.ceil((X.shape[0] - sx) / 2.0)
        oy = np.ceil((X.shape[1] - sy) / 2.0)
        ex = ox + sx
        ey = oy + sy

        return p[ox:ex, oy:ey, :]
    

def GenClippedWidefieldPSF(zs, field_x=0, field_y=0, apertureNA=1.5,
                           apertureZGradient = 0, **kwargs):
    X, Y, R, FP, pupil, u, v = clipped_widefield_pupil_and_propagator(field_x=field_x,
                                                                  field_y=field_y, apertureNA=apertureNA,
                                                                  apertureZGradient = apertureZGradient, **kwargs)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=pupil, zs=zs, **kwargs)

    
def GenZernikePSF(zs, zernikeCoeffs = [], **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    ang = 0
    
    for i, c in enumerate(zernikeCoeffs):
        ang = ang + c*zernike.zernike(i, r, theta)
        
    pupil = F.astype('d')*np.exp(-1j*ang)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=pupil, zs=zs, **kwargs)
    
    
class FourierPSF(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        
    def _pupil(self, F, r, theta):
        """override for derived classes"""
        return F
        
    def e_field(self, zs):
        kwargs = dict(self._kwargs)
        X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

        kwargs.pop('X', None)
        kwargs.pop('Y', None)
        
        theta = np.angle(X + 1j * Y)
        r = R / R[abs(F) > 0].max()
        
        F = self._pupil(F, r, theta)
        
        return E_field_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
    def psf(self, zs):
        return _psf_from_e_field(self.e_field(zs))
        

def GenZernikeDPSF(zs, zernikeCoeffs = {}, **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    kwargs.pop('X', None)
    kwargs.pop('Y', None)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c*zernike.zernike(i, r, theta)
        
    F = F*np.exp(-1j*ang)
        
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)


class ZernikePSF(FourierPSF):
    def __init__(self, zernikeCoeffs = {}, **kwargs):
        FourierPSF.__init__(self, **kwargs)
        
        self.zernikeCoeffs = zernikeCoeffs
        
    def _pupil(self, F, r, theta):
        from PYME.misc import zernike
        ang = 0
        for i, c in self.zernikeCoeffs.items():
            ang = ang + c * zernike.zernike(i, r, theta)

        return F * np.exp(-1j * ang)


def GenZernikeDonutPSF(zs, zernikeCoeffs={}, spiral_amp=1.0, **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    kwargs.pop('X', None)
    kwargs.pop('Y', None)
    
    theta = np.angle(X + 1j * Y)
    r = R / R[abs(F) > 0].max()
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c * zernike.zernike(i, r, theta)
        
    ang = ang + spiral_amp*theta
    
    F = F * np.exp(-1j * ang)
    
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
        
        
    
def GenZernikeDAPSF(zs, zernikeCoeffs = {},field_x=0, field_y=0, apertureNA=1.5,
                    apertureZGradient = 0, **kwargs):
    from PYME.misc import zernike
    
    X, Y, R, FP, F, u, v = clipped_widefield_pupil_and_propagator(field_x=field_x, field_y=field_y,
                                                                  apertureNA=apertureNA,
                                                                  apertureZGradient = apertureZGradient,**kwargs)
    
    theta = np.angle(X + 1j*Y)
    r = R/R[abs(F)>0].max()
    
    ang = 0
    
    for i, c in zernikeCoeffs.items():
        ang = ang + c*zernike.zernike(i, r, theta)
        
    F = F*np.exp(-1j*ang)
        
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)



def GenPRIPSF(zs, strength=1.0, dp=0, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*np.sign(X)*(10*strength*v + dp/2))
    
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)

    
def GenPRIEField(zs, strength=1.0, dp=0, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*np.sign(X)*(10*strength*v + dp/2))

    return EField_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenICPRIPSF(zs, strength=1.0, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator( **kwargs)

    F = F * np.exp(-1j*np.sign(X)*10*strength*v)
    
    F_ = F
    F = F*(X >= 0)
    p1 = PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
    F = F_*(X < 0)
    p2 = PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)

    return  p1 + p2
    
def GenColourPRIPSF(zs, strength=1.0, transmit = [1,1], **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*np.sign(X)*10*strength*v)
    F = F*(np.sqrt(transmit[0])*(X < 0) +  np.sqrt(transmit[1])*(X >=0))

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)

def GenAstigPSF(zs, strength=1.0, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*((strength*v)**2 - 0.5*(strength*R)**2))
    
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenSAPSF(zs, strength=1.0, **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)

    F = F * np.exp(-1j*strength*z8)
    
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenR3PSF(zs, strength=1.0, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = r**3#zernike.zernike(8, r, theta)
    F = F * np.exp(-1j*strength*z8)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenBesselPSF(zs, rad=.95, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    r = R/R[abs(F)>0].max()
    F = F * (r > rad)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenSABesselPSF(zs, rad=.95, strength=1.0, **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)

    F = F * (r > rad)*np.exp(-1j*strength*z8)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)


def GenSAAstigPSF(zs, strength=1.0, SA=0, **kwargs):
    from PYME.misc import zernike
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    r = R/R[abs(F)>0].max()
    theta = np.angle(X + 1j*Y)
    
    z8 = zernike.zernike(8, r, theta)
    a_s = (v**2 - 0.5*R**2)
    
    pf = -(strength*a_s + SA*z8)
    F = F *(np.cos(pf) + j*np.sin(pf))

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    

def GenSAPRIPSF(zs, strength=1.0, SA=0, **kwargs):
    from PYME.misc import zernike
    
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    r = R/R[abs(F)>0].max()

    F = F * np.exp(-1j*np.sign(X)*10*strength*v)
    
    theta = np.angle(X + 1j*Y)
    ang = SA*zernike.zernike(8, r, theta)
            
    F = F*np.exp(-1j*ang)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenDHPSF(zs, vortices=[0.0], **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    
    ph = 0*u
    
    for i, vc in enumerate(vortices):
        sgn = 1#abs(vc)*(i%2)
        ph += sgn*np.angle((u - vc) + 1j*v)

    F = F * np.exp(-1j*ph)

    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenCubicPhasePSF(zs, strength=1.0, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*strength*(u**3 + v**3))
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)

def GenShiftedPSF(zs, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*.01*Y)
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
    
def GenBiplanePSF(zs, zshift = 500, xshift = 1, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)
    F_ = F
    
    F = F * np.exp(-1j*.01*xshift*Y)
    
    ps1 = PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs= zs-zshift/2, **kwargs)
    
    F = F_ * np.exp(1j*.01*xshift*Y)
    
    ps2 = PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs + zshift / 2, **kwargs)
    
    return 0.5*ps1 +0.5*ps2

def GenStripePRIPSF(zs, **kwargs):
    X, Y, R, FP, F, u, v = widefield_pupil_and_propagator(**kwargs)

    F = F * np.exp(-1j*np.sign(np.sin(X))*.005*Y)
    return PSF_from_pupil_and_propagator(X, Y, R, FP, u, v, pupil=F, zs=zs, **kwargs)
   









