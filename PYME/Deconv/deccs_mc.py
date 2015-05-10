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

from pylab import *

from wiener import resizePSF

show_plots = False
from PYME.DSView import View3D

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
        
    def subsearch(self, f0, res, fdef, Afunc, Lfunc, lam2, S):
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
        
        if show_plots:
            figure(1)
            clf()
            subplot(311)
            plot(S)
            subplot(312)
            plot(AS)
            subplot(313)
            plot(LS)
        
        

        Hc = dot(transpose(AS), AS)
        Hw = dot(transpose(LS), LS)
        Gc = dot(transpose(AS), res)
        Gw = dot(transpose(-LS), pref)
        
        #print Hc, Hw, Gc, Gw

        c = solve(Hc + lam2*Hw, Gc + lam2*Gw)
        #print c

        cpred = c0 + dot(dot(transpose(c), Hc), c) - dot(transpose(c), Gc)
        wpred = w0 + dot(dot(transpose(c), Hw), c) - dot(transpose(c), Gw)

        fnew = f0 + dot(S, c) - self.k
        
        if show_plots:
            figure(3)
            clf()
            plot(f0)
            plot(res)
            plot(fnew - f0)
            plot(fnew)

        return (fnew, cpred, wpred)

    def startGuess(self, data):
        '''starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        '''
        return 0*self.g + 1e-3
        


    def deconv(self, data, lamb, num_iters=10, weights=1, k = 0, k2=0):
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
        self.k = k
        self.k2 = k2

        #print data.shape, data.strides
        #print self.Ht.shape, self.Ht.strides

        if 'prep' in dir(self) and not '_F' in dir(self):
            self.prep()

        if not numpy.isscalar(weights):
            self.mask = weights > 0
        else:
            self.mask = numpy.isfinite(data.ravel())

        #guess a starting estimate for the object
        self.f = self.startGuess(data).ravel()
        self.res = 0*data.ravel()

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
        
        lamb2 = lamb*lamb
        
        #v = View3D(self.fs)

        while self.loopcount  < num_iters:
            self.loopcount += 1
            #the direction our prior/ Likelihood function wants us to go
            pref = self.Lfunc(self.f - fdef) + 1e-5

            #the residuals
            #if you want to bodge non-gaussian noise you can multiply with
            #a weighting function - eg: 1/sqrt(data + eps))
            #note that 1/sqrt(data) by itself is not a good idea as it will give
            #infinite weight to zeros. As most devices have some form of readout noise
            #justifying the eps shouldn't be too tricky

            #print data.shape, self.Afunc(self.f).shape
            self.res[:] = (weights*(data - self.Afunc(self.f)))
            #figure(4)
            #worry about the residuals only if they are large
            #plot(self.res)
            self.res[:] = self.res#*(abs(self.res) > 1)#(1 - exp(-(self.res**2)/2))#(abs(self.res) > 1)#
            #self.res[:] = (1 - exp(-(self.res**2)/2))#(abs(self.res) > 1)
            #plot(self.res)
            
            #resulting search directions
            #note that the use of the Likelihood fuction/prior as a search direction
            #is where this method departs from the classical conjugate gradient approach
            S[:,0] = self.Ahfunc(self.res) - self.k2
            #S[:,0] = S[:,0] == S[:,0].max()
            #S[:,0] = S[:,0]*((self.f > 0) + (self.f < 0)*(S[:,0] ==ndimage.maximum_filter(S[:,0].reshape(self.shape), 32).ravel())*(S[:,0] > S[:,0].max()/10))
            S[:,1] = -self.Lhfunc(pref)#/np.maximum(abs(self.res), .1)

            #check to see if the two search directions are orthogonal
            #this can be used as a measure of convergence and a stopping criteria
            test = 1 - abs(dot(S[:,0], S[:,1])/(norm(S[:,0])*norm(S[:,1])))

            #print & log some statistics
            print(('Test Statistic %f' % (test,)))
            self.tests.append(test)
            self.ress.append(norm(self.res))
            self.prefs.append(norm(pref))

            #minimise along search directions to find new estimate
            (fnew, cpred, spred) = self.subsearch(self.f, self.res[self.mask], fdef, self.Afunc, self.Lfunc, lamb2, S[:, 0:nsrch])
            
            S[:,2] = (fnew - self.f)
            
            #try to enforce sparseness by 'sharpening' the resultant image
            fn = fnew
            #if True:#(self.loopcount > 20):# and (self.loopcount % 5):
            #fnew  = fnew - ndimage.gaussian_filter(fnew, 5)#self.Ahfunc(self.Afunc(fnew))
            fnew = fnew - 1#.2*fnew.max()
            fn1 = fnew

            #positivity constraint (not part of original algorithm & could be ommitted)
            fnew = (fnew*(fnew > 1))
            
            #re-normalise
            fnew = fnew*(data.sum()/(fnew.reshape(self.shape).sum(2).ravel()*weights).sum())

            #add last step to search directions, as per classical conj. gradient
            #S[:,2] = (fnew - self.f)
            nsrch = 3

            #set the current estimate to out new estimate
            self.f[:] = fnew
            
            #figure(2)
            #plot(self.f)
            
            if show_plots:
                figure(2)
                clf()
                plot(self.f)
                plot(fn)
                plot(fn1)
                raw_input()
            
            #v.view.Redraw()
            #raw_input()

        return real(self.fs)
        
    def sim_pic(self,data,alpha):
        '''Do the forward transform to simulate a picture. Currently with 4Pi cruft.'''
        self.alpha = alpha
        self.e1 = fftshift(exp(1j*self.alpha))
        self.e2 = fftshift(exp(2j*self.alpha))
        
        return self.Afunc(data)





class dec_conv(dec):
    '''Classical deconvolution with a stationary PSF'''
    lw = 1
    def prep(self):
        #allocate memory
        
        self._F = fftw3f.create_aligned_array(self.FTshape, 'complex64')
        self._r = fftw3f.create_aligned_array(self.shape[:-1], 'f4')

        #calculate plans for other ffts
        self._plan_r_F = fftw3f.Plan(self._r, self._F, 'forward')
        self._plan_F_r = fftw3f.Plan(self._F, self._r, 'backward')


    def psf_calc(self, psf, data_size):
        '''Precalculate the OTF etc...'''
        g = resizePSF(psf, data_size)
        #normalise
        g /= g[:,:,g.shape[2]/2].sum()


        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        self.FTshape = list(self.shape[:-1]) #[self.shape[0], self.shape[1], self.shape[2]/2 + 1]
        self.FTshape[-1] = self.FTshape[-1]/2 +1

        self.g = g.astype('f4');
        
        self.g2 = 1.0*self.g[::-1, ::-1, :]
            
        self.H = []
        self.Ht = []
        
        for i in range(self.shape[-1]):

            #allocate memory
            self.H.append(fftw3f.create_aligned_array(self.FTshape, 'complex64'))
            self.Ht.append(fftw3f.create_aligned_array(self.FTshape, 'complex64'))
            

            #create plans & calculate OTF and conjugate transformed OTF
            fftw3f.Plan(1.0*self.g[:,:,i], self.H[i], 'forward')()
            fftw3f.Plan(1.0*self.g2[:,:,i], self.Ht[i], 'forward')()
    
            self.Ht[i] /= g[:,:,i].size
            self.H[i] /= g[:,:,i].size
    


    def Lfunc(self, f):
        return f#exp(-f) #return sign(f)*(abs(f)**.2 + .001)

    Lhfunc=Lfunc
    
    #def Lfunc(self, f):
    #    return sign(f)*(abs(f)**.2 + .001)

    def Afunc(self, f):
        '''Forward transform - convolve with the PSF'''
        #fs = reshape(f, (self.height, self.width, self.depth))
        fs = f.reshape(self.shape)
        
        r = 0*fs[:,:,0]
        for i in range(self.shape[-1]):
            self._r[:] = fs[:,:,i][:]
    
            #F = fftn(fs)
    
            #d = ifftshift(ifftn(F*self.H));
            self._plan_r_F()
            self._F *= self.H[i]
            self._plan_F_r()
            
            r += ifftshift(self._r)

        #d = real(d);
        return ravel(r)

    def Ahfunc(self, f):
        '''Conjugate transform - convolve with conj. PSF'''
#        fs = reshape(f, (self.height, self.width, self.depth))
#
#        F = fftn(fs)
#        d = ifftshift(ifftn(F*self.Ht));
#        d = real(d);
#        return ravel(d)

        #fs = f.reshape(self.shape[:-1])
        r = np.zeros(self.shape)    
        
        self._r[:] = f.reshape(self._r.shape)

        self._plan_r_F()
        F = self._F.copy()
        for i in range(self.shape[-1]):
            self._F[:] = F*self.Ht[i]#/(self.Ht*self.H + self.lw**2)
            self._plan_F_r()
            r[:,:,i] = ifftshift(self._r)

        return ravel(r)/self.shape[-1]
        


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
        return f

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

#from scipy import ndimage    

def calc_gauss_weights(sigma):
    '''calculate a gaussian filter kernel (adapted from scipy.ndimage.filters.gaussian_filter1d)'''
    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(3.0 * sd + 0.5)
    weights = numpy.zeros(2 * lw + 1, 'float64')
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp

    return weights/sum
    
from scipy.ndimage import _nd_image, _ni_support

class dec_gauss(dec):
    '''Classical deconvolution using non-fft convolution - pot. faster for
    v. small psfs. Note that PSF must be symetric'''
    k = 100
    def psf_calc(self, sigma, data_size, oversamp):
        #g = psf/psf.sum()

        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        #self.g = g
        self.sigma = sigma
        self.kernel = calc_gauss_weights(sigma)
        self.oversamp = oversamp

        #calculate OTF and conjugate transformed OTF
        #self.H = (fftn(g));
        #self.Ht = g.size*(ifftn(g));


    def Lfunc(self, f):
        return f#(f > 0) + 1e-3
        
#    def Lfunc(self, f):
#        return f
#        #return sign(f)*(abs(f)**.2 + .001)
        
#    def Lfunc(self, f):
#        fs = reshape(f, (self.height, self.width, self.depth))
#        a = fs
#    
#        a[:,:,0:-1] += fs[:,:,1:]
#        a[:,:,1:] += fs[:,:,0:-1]
#    
#        a[:,0:-1,:] += fs[:,1:,:]
#        a[:,1:,:] += fs[:,0:-1,:]
#
#        #a[0:-1,:,:] += fs[1:,:,:]
#        #a[1:,:,:] += fs[0:-1,:,:]
#
#        return ravel(cast['f'](a/5.))

#    def Lfunc(self, f):
#        fs = reshape(f, (self.height, self.width, self.depth))
#        
#
#        return ndimage.uniform_filter(fs).ravel()

    Lhfunc=Lfunc

    def Afunc(self, f):
        '''Forward transform - convolve with the PSF'''
        fs = reshape(f, (self.height, self.width, self.depth))

        #d = ndimage.gaussian_filter(fs, self.sigma)
        mode = _ni_support._extend_mode_to_code("reflect")
        #lowpass filter to suppress noise
        #a = ndimage.gaussian_filter(data.astype('f4'), self.filterRadiusLowpass)
        #print data.shape

        output, a = _ni_support._get_output(None, fs)
        _nd_image.correlate1d(fs, self.kernel, 0, output, mode, 0,0)
        _nd_image.correlate1d(output, self.kernel, 1, output, mode, 0,0)
        
        #ndimage.uniform_filter(output, self.oversamp, output=output)

        #d = real(d);
        return ravel(output)#[::oversamp,::oversamp,:])
        
    Ahfunc=Afunc
#    def Ahfunc(self, f):
#        fs = np.zeros((self.height, self.width, self.depth), 'f')
#        fs[::oversamp,::oversamp,:] = f.reshape(self.dataShape)
#        
#        mode = _ni_support._extend_mode_to_code("reflect")
#        #lowpass filter to suppress noise
#        #a = ndimage.gaussian_filter(data.astype('f4'), self.filterRadiusLowpass)
#        #print data.shape
#
#        output, a = _ni_support._get_output(None, fs)
#        _nd_image.correlate1d(fs, self.kernel, 0, output, mode, 0,0)
#        _nd_image.correlate1d(output, self.kernel, 1, output, mode, 0,0)
#        
#        return oversamp*oversamp*output.ravel()

    #def Ahfunc(self, f):
        #'''Conjugate transform - convolve with conj. PSF'''
        #fs = reshape(f, (self.height, self.width, self.depth))

        #d = ndimage.gaussian_filter(fs, self.sigma)
        
        #return ravel(d)
  
        