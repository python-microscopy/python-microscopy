#!/usr/bin/python
##################
# richardsonLucy.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from scipy import *
from scipy.linalg import *
#from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import ndimage
#import dec

class rldec:
    '''Deconvolution class, implementing a variant of the Richardson-Lucy algorithm.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFunc - the likelihood function
    LHFunc - conj. transpose of likelihood function

    see dec_conv for an implementation of conventional image deconvolution with a
    measured, spatially invariant PSF
    '''
    def __init__(self):
       pass

    def startGuess(self, data):
        '''starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        '''
        return data


    def deconv(self, data, lamb, num_iters=10, alpha = None):
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

        #guess a starting estimate for the object
        self.f = self.startGuess(data)

        #make things 1 dimensional
        self.f = self.f.ravel()
        data = data.ravel()

        for loopcount in range(num_iters):
            

            #the residuals
            self.res = data/self.Afunc(self.f);

            print 'Residual norm = %f' % norm(self.res)

            #adjustment
            adjFact = self.Ahfunc(self.res)

            fnew = self.f*adjFact


            #set the current estimate to out new estimate
            self.f = fnew

        return real(self.f)

#class rlbead(rl):
class rlbead(rldec):
    '''Classical deconvolution using non-fft convolution - pot. faster for
    v. small psfs. Note that PSF must be symetric'''
    def psf_calc(self, psf, data_size):
        g = psf#/psf.sum();

        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        self.g = g;

        #calculate OTF and conjugate transformed OTF
        #self.H = (fftn(g));
        #self.Ht = g.size*(ifftn(g));

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