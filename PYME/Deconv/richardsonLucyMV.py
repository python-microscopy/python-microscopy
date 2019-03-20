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
from . import richardsonLucy

class MultiviewRichardsonLucy(richardsonLucy.RichardsonLucyDeconvolution):
    """Deconvolution class, implementing a variant of the Richardson-Lucy algorithm.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    
    This is specifically for multi-view deconvolution, and applies the same forward transform to each view

    see dec_conv for an implementation of conventional image deconvolution with a
    measured, spatially invariant PSF
    """
    
    def deconv(self, views, lamb, num_iters=10, weights = 1, bg = 0):
        """This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter (ignored - kept for compatibility with ICTM)
        num_iters - number of iterations (note that the convergence is fast when
                    compared to many algorithms - e.g Richardson-Lucy - and the
                    default of 10 will usually already give a reasonable result)

        
        """
        #remember what shape we are
        self.dataShape = views[0].shape
        
        if 'prep' in dir(self) and not '_F' in dir(self):
            self.prep()

        if not numpy.isscalar(weights):
            self.mask = 1 - weights #> 0
        else:
            self.mask = 1 - numpy.isfinite(views[0].ravel())

        #guess a starting estimate for the object
        self.f = self.startGuess(np.mean(views, axis=0)).ravel() - bg
        self.fs = self.f.reshape(self.shape)

        #make things 1 dimensional
        #self.f = self.f.ravel()
        views = [d.ravel() for d in views]
        self.views = views
        #data = data.ravel()
        #weights = weights.ravel()
        #print 'dc3'

        mask = 1 - weights
        
        #print data.sum(), self.f.sum()

        self.loopcount=0
        
        

        while self.loopcount  < num_iters:
            self.loopcount += 1
            adjF = 1.0
            for j in range(len(views)):

                #the residuals
                vj = views[j]
                #vj = vj/vj.sum()
                pred =  self.Afunc(self.f)
                #pred = pred/pred.sum()
                self.res = weights*(vj/(pred +1e-12 + 0+ bg)) +  mask;
                
                #adjF *= self.res
                #print vj.sum(), pred.sum()
    
                #adjustment
                adjFact = self.Ahfunc(self.res)
                
                adjF *= adjFact
                #pl.figure()
                #pl.plot(self.res)
                #pl.plot(pred)
                #pl.plot(adjFact)
                
                #pl.figure()
                #pl.plot(adjF)
    
            fnew = self.f*adjF**(1./len(views))
        
            fnew = fnew*self.f.sum()/fnew.sum()


           #set the current estimate to out new estimate
            self.f[:] = fnew
                
                #pl.plot(fnew)
                #print(('Sum = %f' % self.f.sum()))
            
        #print 'dc3'

        return real(self.fs)


class rlbead(MultiviewRichardsonLucy, dec.SpatialConvolutionMapping):
    def __init__(self, *args, **kwargs):
        MultiviewRichardsonLucy.__init__(self, *args, **kwargs)

class dec_conv_slow(MultiviewRichardsonLucy, dec.ClassicMappingNP):
    def __init__(self, *args, **kwargs):
        MultiviewRichardsonLucy.__init__(self, *args, **kwargs)

class dec_conv(MultiviewRichardsonLucy, dec.ClassicMappingFFTW):
    def __init__(self, *args, **kwargs):
        MultiviewRichardsonLucy.__init__(self, *args, **kwargs)