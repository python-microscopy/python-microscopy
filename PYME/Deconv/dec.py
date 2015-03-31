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

from scipy import * 
from scipy.linalg import *
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import ndimage
import fftw3f
import fftwWisdom

fftwWisdom.load_wisdom()
#import weave
#import cDec
#from PYME import pad
import numpy

from wiener import resizePSF

class dec:
    '''Base deconvolution class, implementing a variant of the ICTM algorithm.
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
    '''
    def __init__(self):
        #allocate some empty lists to track our progress in
        self.tests=[]
        self.ress = []
        self.prefs = []
        
    def subsearch(self, f0, res, fdef, Afunc, Lfunc, lam, S):
        '''minimise in subspace - this is the bit which gets called on each iteration
        to work out what the next step is going to be. See Inverse Problems text for details.
        '''
        nsrch = size(S,1)
        pref = Lfunc(f0-fdef)
        w0 = dot(pref, pref)
        c0 = dot(res,res)

        AS = zeros((size(res), nsrch), 'f')
        LS = zeros((size(pref), nsrch), 'f')

        for k in range(nsrch):
            AS[:,k] = Afunc(S[:,k])[self.mask]
            LS[:,k] = Lfunc(S[:,k])

        Hc = dot(transpose(AS), AS)
        Hw = dot(transpose(LS), LS)
        Gc = dot(transpose(AS), res)
        Gw = dot(transpose(-LS), pref)

        c = solve(Hc + pow(lam, 2)*Hw, Gc + pow(lam, 2)*Gw)

        cpred = c0 + dot(dot(transpose(c), Hc), c) - dot(transpose(c), Gc)
        wpred = w0 + dot(dot(transpose(c), Hw), c) - dot(transpose(c), Gw)

        fnew = f0 + dot(S, c)

        return (fnew, cpred, wpred)

    def startGuess(self, data):
        '''starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        '''
        return data.copy()

    def deconvp(self, args):
        ''' convenience function for deconvolving in parallel using processing.Pool.map'''
        self.deconv(*args)
    
    def deconv(self, data, lamb, num_iters=10, weights=1, alpha=None, bg=0):
        '''This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter
        num_iters - number of iterations (note that the convergence is fast when
                    compared to many algorithms - e.g Richardson-Lucy - and the
                    default of 10 will usually already give a reasonable result)

        alpha - PSF phase - hacked in for variable phase 4Pi deconvolution, should
                really be refactored out into the dec_4pi classes.
        '''
        #remember what shape we are
        self.dataShape = data.shape

        #print data.shape, data.strides
        #print self.Ht.shape, self.Ht.strides

        if 'prep' in dir(self) and not '_F' in dir(self):
            self.prep()

        if not numpy.isscalar(weights):
            self.mask = weights > 0
        else:
            self.mask = numpy.isfinite(data.ravel())

        #if doing 4Pi dec, do some phase related precomputation
        #TODO move this to the 4Pi_dec classes
        if (not alpha == None):
            self.alpha = alpha
            self.e1 = fftshift(exp(1j*self.alpha))
            self.e2 = fftshift(exp(2j*self.alpha))

        #guess a starting estimate for the object
        self.f = self.startGuess(data).ravel()
        self.res = 0*self.f

        self.fs = self.f.reshape(self.shape)

        #make things 1 dimensional
        #self.f = self.f.ravel()
        data = data.ravel()

        #print data.mean(), weights, lamb
        #print abs(self.H).sum()
        #print abs(self.Ht).sum()

        #use 0 as the default solution - should probably be refactored like the starting guess
        fdef = zeros(self.f.shape, 'f')

        #initial search directions
        S = zeros((size(self.f), 3), 'f')

        #number of search directions
        nsrch = 2
        self.loopcount = 0

        while self.loopcount  < num_iters:
            self.loopcount += 1
            #the direction our prior/ Likelihood function wants us to go
            pref = self.Lfunc(self.f - fdef);

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

            #check to see if the two search directions are orthogonal
            #this can be used as a measure of convergence and a stopping criteria
            test = 1 - abs(dot(S[:,0], S[:,1])/(norm(S[:,0])*norm(S[:,1])))

            #print & log some statistics
            print(('Test Statistic %f' % (test,)))
            self.tests.append(test)
            self.ress.append(norm(self.res))
            self.prefs.append(norm(pref))

            #minimise along search directions to find new estimate
            (fnew, cpred, spred) = self.subsearch(self.f, self.res[self.mask], fdef, self.Afunc, self.Lfunc, lamb, S[:, 0:nsrch])

            #positivity constraint (not part of original algorithm & could be ommitted)
            fnew = (fnew*(fnew > 0))

            #add last step to search directions, as per classical conj. gradient
            S[:,2] = (fnew - self.f)
            nsrch = 3

            #set the current estimate to out new estimate
            self.f[:] = fnew

        return real(self.fs)
        
    def sim_pic(self,data,alpha):
        '''Do the forward transform to simulate a picture. Currently with 4Pi cruft.'''
        self.alpha = alpha
        self.e1 = fftshift(exp(1j*self.alpha))
        self.e2 = fftshift(exp(2j*self.alpha))
        
        return self.Afunc(data)



class dec_4pi(dec):
    '''Variable phase 4Pi deconvolution'''
    def psf_calc(self, psf, kz, data_size):
        '''Pre calculate OTFs etc ...'''
        g = psf;

        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        (x,y,z) = mgrid[-floor(self.height/2.0):(ceil(self.height/2.0)), -floor(self.width/2.0):(ceil(self.width/2.0)), -floor(self.depth/2.0):(ceil(self.depth/2.0))]
        
       
        gs = shape(g);
        g = g[int(floor((gs[0] - self.height)/2)):int(self.height + floor((gs[0] - self.height)/2)), int(floor((gs[1] - self.width)/2)):int(self.width + floor((gs[1] - self.width)/2)), int(floor((gs[2] - self.depth)/2)):int(self.depth + floor((gs[2] - self.depth)/2))]
	
        g = abs(ifftshift(ifftn(abs(fftn(g)))));
        g = (g/sum(sum(sum(g))));
	
        self.g = g;
        
        self.H = cast['f'](fftn(g));
        self.Ht = cast['f'](ifftn(g));

        tk = 2*kz*z
        
        t = g*exp(1j*tk)
        self.He = cast['F'](fftn(t));    
        self.Het = cast['F'](ifftn(t));

        tk = 2*tk
        
        t = g*exp(1j*tk)
        self.He2 = cast['F'](fftn(t));    
        self.He2t = cast['F'](ifftn(t));

        

    def Lfunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))
        a = -6*fs
    
        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]
    
        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        return ravel(cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))

        F = fftn(fs)

        d_1 = ifftshift(ifftn(F*self.H));
        
        d_e = ifftshift(ifftn(F*self.He));
        
        d_e2 = ifftshift(ifftn(F*self.He2));
        
        d = (1.5*real(d_1) + 2*real(d_e*self.e1) + 0.5*real(d_e2*self.e2))
        
        d = real(d);
        return ravel(d)

    def Ahfunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))

        F = fftn(fs)

        d_1 = ifftshift(ifftn(F*self.Ht));

        d_e = ifftshift(ifftn(F*self.Het));

        d_e2 = ifftshift(ifftn(F*self.He2t));

        d = (1.5*d_1 + 2*real(d_e*exp(1j*self.alpha)) + 0.5*real(d_e2*exp(2*1j*self.alpha)));
        
        d = real(d);
        return ravel(d)

class dec_conv(dec):
    '''Classical deconvolution with a stationary PSF'''
    def prep(self):
        #allocate memory
        
        self._F = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        self._r = fftw3f.create_aligned_array(self.shape, 'f4')

        #calculate plans for other ffts
        self._plan_r_F = fftw3f.Plan(self._r, self._F, 'forward')
        self._plan_F_r = fftw3f.Plan(self._F, self._r, 'backward')


    def psf_calc(self, psf, data_size):
        '''Precalculate the OTF etc...'''
#        pw = (numpy.array(data_size) - psf.shape)/2.
#        pw1 = numpy.floor(pw)
#        pw2 = numpy.ceil(pw)
#
#        g = psf/psf.sum()
#
#        #work out how we're going to need to pad to get the PSF the same size as our data
#        if pw1[0] < 0:
#            if pw2[0] < 0:
#                g = g[-pw1[0]:pw2[0]]
#            else:
#                g = g[-pw1[0]:]
#
#            pw1[0] = 0
#            pw2[0] = 0
#
#        if pw1[1] < 0:
#            if pw2[1] < 0:
#                g = g[-pw1[1]:pw2[1]]
#            else:
#                g = g[-pw1[1]:]
#
#            pw1[1] = 0
#            pw2[1] = 0
#
#        if pw1[2] < 0:
#            if pw2[2] < 0:
#                g = g[-pw1[2]:pw2[2]]
#            else:
#                g = g[-pw1[2]:]
#
#            pw1[2] = 0
#            pw2[2] = 0
#
#
#        #do the padding
#        #g = pad.with_constant(g, ((pw2[0], pw1[0]), (pw2[1], pw1[1]),(pw2[2], pw1[2])), (0,))
#
#        g_ = fftw3f.create_aligned_array(data_size, 'float32')
#        g_[:] = 0
#        #print g.shape, g_.shape, g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:-pw1[2]].shape
#        if pw1[2] == 0:
#            g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:] = g
#        else:
#            g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:-pw1[2]] = g
#        g = g_
        g = resizePSF(psf, data_size)


        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        self.FTshape = [self.shape[0], self.shape[1], self.shape[2]/2 + 1]

        self.g = g.astype('f4');
        self.g2 = 1.0*self.g[::-1, ::-1, ::-1]

        #allocate memory
        self.H = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        self.Ht = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        #self.f = zeros(self.shape, 'f4')
        #self.res = zeros(self.shape, 'f4')
        #self.S = zeros((size(self.f), 3), 'f4')

        #self._F = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        #self._r = fftw3f.create_aligned_array(self.shape, 'f4')
        #S0 = self.S[:,0]

        #create plans & calculate OTF and conjugate transformed OTF
        fftw3f.Plan(self.g, self.H, 'forward')()
        fftw3f.Plan(self.g2, self.Ht, 'forward')()

        self.Ht /= g.size;
        self.H /= g.size;

        #calculate plans for other ffts
        #self._plan_r_F = fftw3f.Plan(self._r, self._F, 'forward')
        #self._plan_F_r = fftw3f.Plan(self._F, self._r, 'backward')


    def Lfunc(self, f):
        '''convolve with an approximate 2nd derivative likelihood operator in 3D.
        i.e. [[[0,0,0][0,1,0][0,0,0]],[[0,1,0][1,-6,1][0,1,0]],[[0,0,0][0,1,0][0,0,0]]]
        '''
        #make our data 3D again
        fs = reshape(f, (self.height, self.width, self.depth))
        a = -6*fs

        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]

        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        #flatten data again
        return ravel(cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        '''Forward transform - convolve with the PSF'''
        #fs = reshape(f, (self.height, self.width, self.depth))
        self._r[:] = f.reshape(self._r.shape)

        #F = fftn(fs)

        #d = ifftshift(ifftn(F*self.H));
        self._plan_r_F()
        self._F *= self.H
        self._plan_F_r()

        #d = real(d);
        return ravel(ifftshift(self._r))

    def Ahfunc(self, f):
        '''Conjugate transform - convolve with conj. PSF'''
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

        return ravel(ifftshift(self._r))

class dec_bead(dec):
    '''Classical deconvolution using non-fft convolution - pot. faster for
    v. small psfs. Note that PSF must be symetric'''
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
        '''convolve with an approximate 2nd derivative likelihood operator in 3D.
        i.e. [[[0,0,0][0,1,0][0,0,0]],[[0,1,0][1,-6,1][0,1,0]],[[0,0,0][0,1,0][0,0,0]]]
        '''
        #make our data 3D again
        fs = reshape(f, (self.height, self.width, self.depth))
        a = -6*fs

        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]

        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        #flatten data again
        return ravel(cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        '''Forward transform - convolve with the PSF'''
        fs = reshape(f, (self.height, self.width, self.depth))

        d = ndimage.convolve(fs, self.g)

        #d = real(d);
        return ravel(d)

    def Ahfunc(self, f):
        '''Conjugate transform - convolve with conj. PSF'''
        fs = reshape(f, (self.height, self.width, self.depth))

        d = ndimage.correlate(fs, self.g)
        
        return ravel(d)
    
class dec_4pi_c(dec_4pi):
    def prepare(self):
        return cDec.prepare(shape(self.H))
    
    def cleanup(self):
        return cDec.cleanup()
    
    def Afunc(self, f):
        return ravel(ifftshift(reshape(cDec.fw_map(cast['F'](f),cast['F'](self.alpha), cast['F'](self.H), cast['F'](self.He), cast['F'](self.He2), cast['F'](self.e1), cast['F'](self.e2)), shape(self.alpha))))
    
    def Ahfunc(self, f):
        #return cDec.fw_map(f,self.alpha, self.Ht, self.Het, self.He2t, self.e1, self.e2)
        return ravel(ifftshift(reshape(cDec.fw_map(cast['F'](f),cast['F'](self.alpha), cast['F'](self.Ht), cast['F'](self.Het), cast['F'](self.He2t), cast['F'](self.e1), cast['F'](self.e2)), shape(self.alpha))))
    
    def Lfunc(self,f):
        return cDec.Lfunc(f, shape(self.alpha))
    
    Lhfunc = Lfunc
  
        