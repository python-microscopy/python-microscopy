#!/usr/bin/python

##################
# dec.py
#
# Copyright David Baddeley, 2009
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

from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import ndimage
import numpy as np

import PYME.misc.fftw_compat as fftw3f
from .wiener import resizePSF

from . import fftwWisdom
fftwWisdom.load_wisdom()

NTHREADS = 8
FFTWFLAGS = ['measure']



class ICTMDeconvolution(object):
    """Base deconvolution class, implementing a variant of the ICTM algorithm.
    ie. find f such that:
       ||Af-d||^2 + lamb^2||L(f - fdef)||^2
    is minimised
    
    Note that this is nominally for Gaussian distributed noise, although can be
    adapted by adding a weighting to the misfit term.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFunc - the likelihood function
    LHFunc - conj. transpose of likelihood function

    see dec_conv for an implementation of conventional image deconvolution with a
    measured, spatially invariant PSF
    """
    def __init__(self):
        #allocate some empty lists to track our progress in
        self.tests=[]
        self.ress = []
        self.prefs = []
        
    def subsearch(self, f0, res, fdef, Afunc, Lfunc, lam, S):
        """minimise in subspace - this is the bit which gets called on each iteration
        to work out what the next step is going to be. See Inverse Problems text for details.
        """
        nsrch = np.size(S,1)
        pref = Lfunc(f0-fdef)
        w0 = np.dot(pref, pref)
        c0 = np.dot(res,res)

        AS = np.zeros((np.size(res), nsrch), 'f')
        LS = np.zeros((np.size(pref), nsrch), 'f')

        for k in range(nsrch):
            AS[:,k] = Afunc(S[:,k])[self.mask]
            LS[:,k] = Lfunc(S[:,k])

        Hc = np.dot(np.transpose(AS), AS)
        Hw = np.dot(np.transpose(LS), LS)
        Gc = np.dot(np.transpose(AS), res)
        Gw = np.dot(np.transpose(-LS), pref)

        c = np.linalg.solve(Hc + pow(lam, 2)*Hw, Gc + pow(lam, 2)*Gw)

        cpred = c0 + np.dot(np.dot(np.transpose(c), Hc), c) - np.dot(np.transpose(c), Gc)
        wpred = w0 + np.dot(np.dot(np.transpose(c), Hw), c) - np.dot(np.transpose(c), Gw)

        fnew = f0 + np.dot(S, c)

        return (fnew, cpred, wpred)

    def startGuess(self, data):
        """starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        """
        return data.copy()

    def deconvp(self, args):
        """ convenience function for deconvolving in parallel using processing.Pool.map"""
        self.deconv(*args)
    
    def deconv(self, data, lamb, num_iters=10, weights=1, alpha=None, bg=0, pos=True):
        """This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter
        num_iters - number of iterations (note that the convergence is fast when
                    compared to many algorithms - e.g Richardson-Lucy - and the
                    default of 10 will usually already give a reasonable result)
        weights - a weighting on the residuals to deweight missing data and/or adjust the
                  noise behaviour
        alpha - PSF phase - hacked in for variable phase 4Pi deconvolution, should
                really be refactored out into the dec_4pi classes.
        bg - Dummy variable
        pos - Flag to turn positivity constraints on/off
        """
        #remember what shape we are
        self.dataShape = data.shape
        
        if 'prep' in dir(self) and not '_F' in dir(self):
            self.prep()

        if not np.isscalar(weights):
            self.mask = weights > 0
        else:
            self.mask = np.isfinite(data.ravel())

        #if doing 4Pi dec, do some phase related precomputation
        #TODO move this to the 4Pi_dec classes
        if (not alpha is None):
            self.alpha = alpha
            self.e1 = fftshift(np.exp(1j*self.alpha))
            self.e2 = fftshift(np.exp(2j*self.alpha))

        #guess a starting estimate for the object
        self.f = self.startGuess(data).ravel()

        self.fs = self.f.reshape(self.shape)

        #make things 1 dimensional
        #self.f = self.f.ravel()
        data = data.ravel()
        self.res = 0*data
        
        if not np.isscalar(weights):
            weights = weights / weights.mean()

        #use 0 as the default solution - should probably be refactored like the starting guess
        fdef = np.zeros(self.f.shape, 'f')

        #initial search directions
        S = np.zeros((np.size(self.f), 3), 'f')

        #number of search directions
        nsrch = 2
        self.loopcount = 0

        while (self.loopcount  < num_iters) and (not self._stop_cond()):
            self.loopcount += 1
            #the direction our prior/ Likelihood function wants us to go
            pref = self.Lfunc(self.f - fdef)
            
            #print pref, self.f

            #the residuals
            #if you want to bodge non-gaussian noise you can multiply with
            #a weighting function - eg: 1/sqrt(data + eps))
            #note that 1/sqrt(data) by itself is not a good idea as it will give
            #infinite weight to zeros. As most devices have some form of readout noise
            #justifying the eps shouldn't be too tricky
            self.res[:] = (weights*(data - self.Afunc(self.f)))
            
            #resulting search directions
            #note that the use of the Likelihood fuction/prior as a search direction
            #is where this method departs from the classical conjugate gradient approach
            S[:,0] = self.Ahfunc(self.res)
            S[:,1] = -self.Lhfunc(pref)
            
            #print norm(S[:,0]), norm(S[:,1])

            #check to see if the two search directions are orthogonal
            #this can be used as a measure of convergence and a stopping criteria
            test = 1 - abs(np.dot(S[:,0], S[:,1])/(np.linalg.norm(S[:,0])*np.linalg.norm(S[:,1])))

            #print & log some statistics
            print(('Test Statistic %f' % (test,)))
            self.tests.append(test)
            self.ress.append(np.linalg.norm(self.res))
            self.prefs.append(np.linalg.norm(pref))

            #minimise along search directions to find new estimate
            (fnew, cpred, spred) = self.subsearch(self.f, self.res[self.mask], fdef, self.Afunc, self.Lfunc, lamb, S[:, 0:nsrch])

            #positivity constraint (not part of original algorithm & could be ommitted)
            
            if pos:
                fnew = (fnew*(fnew > 0))

            #add last step to search directions, as per classical conj. gradient
            S[:,2] = (fnew - self.f)
            nsrch = 3

            #set the current estimate to out new estimate
            self.f[:] = fnew

        return np.real(self.fs)

    def _stop_cond(self):
        """Optional stopping condition to end deconvolution early."""
        return False
        
    def sim_pic(self,data,alpha):
        """Do the forward transform to simulate a picture. Currently with 4Pi cruft."""
        self.alpha = alpha
        self.e1 = fftshift(np.exp(1j*self.alpha))
        self.e2 = fftshift(np.exp(2j*self.alpha))
        
        return self.Afunc(data)


class DeconvMappingBase(object):
    """
    Base class for different types of deconvolution. Ultimate deconvolution classes will inherit from both this and a
    a method class (e.g. ICTM Deconvolution)
    
    Provides implementations of the following:
    
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFunc - the likelihood function
    LHFunc - conj. transpose of likelihood function
    
    Also, defines:
    
    psf_calc - does any pre-computation to get the PSF into a usable form
    prep - allocate memory / setup FFTW plans etc ...
    
    """
    
    def Afunc(self, f):
        """ The forward mapping"""
        raise NotImplementedError('Must be over-ridden in derived class')
    
    def Ahfunc(self, f):
        """ The conjugate transpose of the forward mapping"""
        raise NotImplementedError('Must be over-ridden in derived class')
    
    def Lfunc(self, f):
        """ The likelihood function (ICTM deconvolution only)"""
        raise NotImplementedError('Must be over-ridden in derived class')
    
    def Lhfunc(self, f):
        """ The gonjugate transpose of likelihood function (ICTM deconvolution only)"""
        raise NotImplementedError('Must be over-ridden in derived class')
    
    def psf_calc(self, psf, data_size):
        """
        do any pre-computation on the PSF. e.g. resizing to match the data shape and/or pre-calculating and
        storing the OTF
        """
        raise NotImplementedError('Must be over-ridden in derived class')
    
    def prep(self):
        """
        Allocate memory and compute FFTW plans etc (this is separate from psf_calc as an aid to distributing Deconvolution
        objects for parallel processing - psf_calc gets called before Deconv objects are passed, prep gets called when
        the deconvolution is run.

        """

class ClassicMappingNP(DeconvMappingBase):
    """Classical deconvolution with a stationary PSF - uses numpy FTTs rather than FFTW"""
    def psf_calc(self, psf, data_size):
        """Precalculate the OTF etc..."""

        g = resizePSF(psf, data_size)

        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        self.g = g

        #calculate OTF and conjugate transformed OTF
        self.H = (fftn(g))
        self.Ht = g.size*(ifftn(g))


    def Afunc(self, f):
        """Forward transform - convolve with the PSF"""
        fs = np.reshape(f, (self.height, self.width, self.depth))

        F = fftn(fs)

        d = ifftshift(ifftn(F*self.H))

        d = np.real(d)
        return np.ravel(d)

    def Ahfunc(self, f):
        """Conjugate transform - convolve with conj. PSF"""
        fs = np.reshape(f, (self.height, self.width, self.depth))

        F = fftn(fs)
        d = ifftshift(ifftn(F*self.Ht))
        d = np.real(d)
        return np.ravel(d)
    
    def Lfunc(self, f):
        """convolve with an approximate 2nd derivative likelihood operator in 3D.
        i.e. [[[0,0,0][0,1,0][0,0,0]],[[0,1,0][1,-6,1][0,1,0]],[[0,0,0][0,1,0][0,0,0]]]
        """
        #make our data 3D again
        fs = np.reshape(f, (self.height, self.width, self.depth))
        a = -6*fs

        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]

        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        #flatten data again
        return np.ravel(np.cast['f'](a))

    Lhfunc=Lfunc

class dec_conv_slow(ICTMDeconvolution, ClassicMappingNP):
    def __init__(self, *args, **kwargs):
        ICTMDeconvolution.__init__(self, *args, **kwargs)

class ClassicMappingFFTW(DeconvMappingBase):
    """Classical deconvolution with a stationary PSF using FFTW for convolutions"""
    def prep(self):
        #allocate memory
        self._F = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        self._r = fftw3f.create_aligned_array(self.shape, 'f4')

        print('Creating plans for FFTs - this might take a while')
        #calculate plans for other ffts
        self._plan_r_F = fftw3f.Plan(self._r, self._F, 'forward', flags = FFTWFLAGS, nthreads=NTHREADS)
        self._plan_F_r = fftw3f.Plan(self._F, self._r, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()

        print('Done planning')
    
    def psf_calc(self, psf, data_size):
        """Precalculate the OTF etc..."""
        
        g = resizePSF(psf, data_size)

        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size
        
        print('Calculating OTF')

        self.FTshape = [self.shape[0], self.shape[1], int(self.shape[2]/2 + 1)]

        self.g = g.astype('f4')
        self.g2 = 1.0*self.g[::-1, ::-1, ::-1]

        #allocate memory
        self.H = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        self.Ht = fftw3f.create_aligned_array(self.FTshape, 'complex64')

        #create plans & calculate OTF and conjugate transformed OTF
        fftw3f.Plan(self.g, self.H, 'forward')()
        fftw3f.Plan(self.g2, self.Ht, 'forward')()

        self.Ht /= g.size
        self.H /= g.size




    def Lfunc(self, f):
        """convolve with an approximate 2nd derivative likelihood operator in 3D.
        i.e. [[[0,0,0][0,1,0][0,0,0]],[[0,1,0][1,-6,1][0,1,0]],[[0,0,0][0,1,0][0,0,0]]]
        """
        #make our data 3D again
        fs = np.reshape(f, (self.height, self.width, self.depth))
        a = -6*fs

        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]

        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        #flatten data again
        return np.ravel(np.cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        """Forward transform - convolve with the PSF"""
        #fs = reshape(f, (self.height, self.width, self.depth))
        self._r[:] = f.reshape(self._r.shape)

        #F = fftn(fs)

        #d = ifftshift(ifftn(F*self.H));
        self._plan_r_F()
        self._F *= self.H
        self._plan_F_r()

        #d = real(d);
        return np.ravel(ifftshift(self._r))

    def Ahfunc(self, f):
        """Conjugate transform - convolve with conj. PSF"""
#        fs = reshape(f, (self.height, self.width, self.depth))
#
#        F = fftn(fs)
#        d = ifftshift(ifftn(F*self.Ht));
#        d = real(d);
#        return ravel(d)
        self._r[:] = f.reshape(self._r.shape)

        self._plan_r_F()
        self._F *= self.Ht
        self._plan_F_r()

        return np.ravel(ifftshift(self._r))

class dec_conv(ICTMDeconvolution, ClassicMappingFFTW):
    def __init__(self, *args, **kwargs):
        ICTMDeconvolution.__init__(self, *args, **kwargs)

class SpatialConvolutionMapping(DeconvMappingBase):
    """Classical deconvolution using non-fft convolution - pot. faster for
    v. small psfs. Note that PSF must be symetric"""
    def psf_calc(self, psf, data_size):
        g = psf/psf.sum()

        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        self.g = g

        #calculate OTF and conjugate transformed OTF
        #self.H = (fftn(g));
        #self.Ht = g.size*(ifftn(g));


    def Lfunc(self, f):
        """convolve with an approximate 2nd derivative likelihood operator in 3D.
        i.e. [[[0,0,0][0,1,0][0,0,0]],[[0,1,0][1,-6,1][0,1,0]],[[0,0,0][0,1,0][0,0,0]]]
        """
        #make our data 3D again
        fs = np.reshape(f, (self.height, self.width, self.depth))
        a = -6*fs

        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]

        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        #flatten data again
        return np.ravel(np.cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        """Forward transform - convolve with the PSF"""
        fs = np.reshape(f, (self.height, self.width, self.depth))

        d = ndimage.convolve(fs, self.g)

        #d = real(d);
        return np.ravel(d)

    def Ahfunc(self, f):
        """Conjugate transform - convolve with conj. PSF"""
        fs = np.reshape(f, (self.height, self.width, self.depth))

        d = ndimage.correlate(fs, self.g)
        
        return np.ravel(d)

class dec_bead(ICTMDeconvolution, SpatialConvolutionMapping):
    def __init__(self, *args, **kwargs):
        ICTMDeconvolution.__init__(self, *args, **kwargs)

class dec_4pi(ICTMDeconvolution):
    """Variable phase 4Pi deconvolution"""
    
    def psf_calc(self, psf, kz, data_size):
        """Pre calculate OTFs etc ..."""
        g = psf;
        
        self.height = data_size[0]
        self.width = data_size[1]
        self.depth = data_size[2]
        
        (x, y, z) = np.mgrid[-np.floor(self.height / 2.0):(np.ceil(self.height / 2.0)),
                    -np.floor(self.width / 2.0):(np.ceil(self.width / 2.0)),
                    -np.floor(self.depth / 2.0):(np.ceil(self.depth / 2.0))]
        
        gs = np.shape(g)
        g = g[int(np.floor((gs[0] - self.height) / 2)):int(self.height + np.floor((gs[0] - self.height) / 2)),
            int(np.floor((gs[1] - self.width) / 2)):int(self.width + np.floor((gs[1] - self.width) / 2)),
            int(np.floor((gs[2] - self.depth) / 2)):int(self.depth + np.floor((gs[2] - self.depth) / 2))]
        
        g = abs(ifftshift(ifftn(abs(fftn(g)))))
        g = (g / sum(sum(sum(g))))
        
        self.g = g;
        
        self.H = fftn(g).astype('f')
        self.Ht = ifftn(g).astype('f')
        
        tk = 2 * kz * z
        
        t = g * np.exp(1j * tk)
        self.He = np.cast['F'](fftn(t));
        self.Het = np.cast['F'](ifftn(t));
        
        tk = 2 * tk
        
        t = g * np.exp(1j * tk)
        self.He2 = np.cast['F'](fftn(t));
        self.He2t = np.cast['F'](ifftn(t));
    
    def Lfunc(self, f):
        fs = np.reshape(f, (self.height, self.width, self.depth))
        a = -6 * fs
        
        a[:, :, 0:-1] += fs[:, :, 1:]
        a[:, :, 1:] += fs[:, :, 0:-1]
        
        a[:, 0:-1, :] += fs[:, 1:, :]
        a[:, 1:, :] += fs[:, 0:-1, :]
        
        a[0:-1, :, :] += fs[1:, :, :]
        a[1:, :, :] += fs[0:-1, :, :]
        
        return np.ravel(np.cast['f'](a))
    
    Lhfunc = Lfunc
    
    def Afunc(self, f):
        fs = np.reshape(f, (self.height, self.width, self.depth))
        
        F = fftn(fs)
        
        d_1 = ifftshift(ifftn(F * self.H));
        
        d_e = ifftshift(ifftn(F * self.He));
        
        d_e2 = ifftshift(ifftn(F * self.He2));
        
        d = (1.5 * np.real(d_1) + 2 * np.real(d_e * self.e1) + 0.5 * np.real(d_e2 * self.e2))
        
        d = np.real(d);
        return np.ravel(d)
    
    def Ahfunc(self, f):
        fs = np.reshape(f, (self.height, self.width, self.depth))
        
        F = fftn(fs)
        
        d_1 = ifftshift(ifftn(F * self.Ht));
        
        d_e = ifftshift(ifftn(F * self.Het));
        
        d_e2 = ifftshift(ifftn(F * self.He2t));
        
        d = (1.5 * d_1 + 2 * np.real(d_e * np.exp(1j * self.alpha)) + 0.5 * np.real(d_e2 * np.exp(2 * 1j * self.alpha)));
        
        d = np.real(d);
        return np.ravel(d)


class dec_4pi_c(dec_4pi):
    def prepare(self):
        return cDec.prepare(np.shape(self.H))
    
    def cleanup(self):
        return cDec.cleanup()
    
    def Afunc(self, f):
        return np.ravel(ifftshift(np.reshape(cDec.fw_map(np.cast['F'](f), np.cast['F'](self.alpha), np.cast['F'](self.H), np.cast['F'](self.He), np.cast['F'](self.He2), np.cast['F'](self.e1), np.cast['F'](self.e2)), np.shape(self.alpha))))
    
    def Ahfunc(self, f):
        #return cDec.fw_map(f,self.alpha, self.Ht, self.Het, self.He2t, self.e1, self.e2)
        return np.ravel(ifftshift(np.reshape(cDec.fw_map(np.cast['F'](f), np.cast['F'](self.alpha), np.cast['F'](self.Ht), np.cast['F'](self.Het), np.cast['F'](self.He2t), np.cast['F'](self.e1), np.cast['F'](self.e2)), np.shape(self.alpha))))
    
    def Lfunc(self,f):
        return cDec.Lfunc(f, np.shape(self.alpha))
    
    Lhfunc = Lfunc
  
        