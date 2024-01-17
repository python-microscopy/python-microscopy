#!/usr/bin/python
##################
# richardsonLucy.py
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

from scipy import *
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import ndimage

import numpy
import numpy as np

import PYME.misc.fftw_compat as fftw3f
from .wiener import resizePSF

from . import fftwWisdom
fftwWisdom.load_wisdom()

NTHREADS = 8
FFTWFLAGS = ['measure']

from . import dec

class RichardsonLucyDeconvolution(object):
    """Deconvolution class, implementing a variant of the Richardson-Lucy algorithm.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFunc - the likelihood function
    LHFunc - conj. transpose of likelihood function

    see dec_conv for an implementation of conventional image deconvolution with a
    measured, spatially invariant PSF
    """
    def __init__(self):
       pass

    def startGuess(self, data):
        """starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        """
        return 0*data + data.mean()


    def deconvp(self, args):
        """ convenience function for deconvolving in parallel using processing.Pool.map"""
        return self.deconv(*args)
        #return 0
    
    def deconv(self, data, lamb, num_iters=10, weights = 1, bg = 0):
        """This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter (ignored - kept for compatibility with ICTM)
        num_iters - number of iterations (note that the convergence is fast when
                    compared to many algorithms - e.g Richardson-Lucy - and the
                    default of 10 will usually already give a reasonable result)

        
        """
        #remember what shape we are
        self.dataShape = data.shape
        
        if 'prep' in dir(self) and not '_F' in dir(self):
            self.prep()

        if not numpy.isscalar(weights):
            self.mask = 1 - weights #> 0
        else:
            self.mask = 1 - numpy.isfinite(data.ravel())

        #guess a starting estimate for the object
        self.f = self.startGuess(data).ravel() - bg
        self.fs = self.f.reshape(self.shape)

        #make things 1 dimensional
        #self.f = self.f.ravel()
        data = data.ravel()

        self.loopcount=0

        while self.loopcount  < num_iters:
            self.loopcount += 1

            #the residuals
            self.res = weights*(data/(self.Afunc(self.f)+1e-12 + bg)) +  self.mask

            #self.res = weights*(data/(ndimage.median_filter(self.Afunc(self.f), 3)+1e-12 + bg)) +  mask;
            #self.res = weights*(data/(self.Afunc(self.f)+1e-12 + bg)) +  mask;

            #self.res = (data/(self.Afunc(self.f) + 1e-12))

            #print 'Residual norm = %f' % norm(self.res)
            

            #adjustment
            adjFact = self.Ahfunc(self.res)
            
            #adjFact*= .5 + ndimage.median_filter(self.res, 3)
            
            #adjFact/= adjFact.mean()
            #print self.res.min(), adjFact.min()

            fnew = self.f*adjFact

            #set the current estimate to out new estimate
            self.f[:] = fnew

        return np.real(self.fs)


class rlbead(RichardsonLucyDeconvolution, dec.SpatialConvolutionMapping):
    def __init__(self, *args, **kwargs):
        RichardsonLucyDeconvolution.__init__(self, *args, **kwargs)

class dec_conv_slow(RichardsonLucyDeconvolution, dec.ClassicMappingNP):
    def __init__(self, *args, **kwargs):
        RichardsonLucyDeconvolution.__init__(self, *args, **kwargs)

class dec_conv(RichardsonLucyDeconvolution, dec.ClassicMappingFFTW):
    def __init__(self, *args, **kwargs):
        RichardsonLucyDeconvolution.__init__(self, *args, **kwargs)