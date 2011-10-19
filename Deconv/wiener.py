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
#from scipy import *
#from scipy.linalg import *
#from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
#from scipy import ndimage
import numpy
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
import fftw3f
import fftwWisdom

fftwWisdom.load_wisdom()

#import weave
#import cDec
#from PYME import pad
#import dec

NTHREADS = 1
FFTWFLAGS = ['measure']


class dec_wiener:
    '''Classical deconvolution with a stationary PSF'''
    def psf_calc(self, psf, data_size):
        '''Precalculate the OTF etc...'''
        pw = (numpy.array(data_size) - psf.shape)/2.
        pw1 = numpy.floor(pw)
        pw2 = numpy.ceil(pw)

        g = psf/psf.sum()

        #work out how we're going to need to pad to get the PSF the same size as our data
        if pw1[0] < 0:
            if pw2[0] < 0:
                g = g[-pw1[0]:pw2[0]]
            else:
                g = g[-pw1[0]:]

            pw1[0] = 0
            pw2[0] = 0

        if pw1[1] < 0:
            if pw2[1] < 0:
                g = g[-pw1[1]:pw2[1]]
            else:
                g = g[-pw1[1]:]

            pw1[1] = 0
            pw2[1] = 0

        if pw1[2] < 0:
            if pw2[2] < 0:
                g = g[-pw1[2]:pw2[2]]
            else:
                g = g[-pw1[2]:]

            pw1[2] = 0
            pw2[2] = 0


        #do the padding
        #g = pad.with_constant(g, ((pw2[0], pw1[0]), (pw2[1], pw1[1]),(pw2[2], pw1[2])), (0,))
        g_ = fftw3f.create_aligned_array(data_size, 'float32')
        g_[:] = 0
        #print g.shape, g_.shape, g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:-pw1[2]].shape
        if pw1[2] == 0:
            g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:] = g
        else:
            g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:-pw1[2]] = g
        #g_[pw2[0]:-pw1[0], pw2[1]:-pw1[1], pw2[2]:-pw1[2]] = g
        g = g_


        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        FTshape = [self.shape[0], self.shape[1], self.shape[2]/2 + 1]

        self.g = g.astype('f4');
        self.g2 = 1.0*self.g[::-1, ::-1, ::-1]

        #allocate memory
        self.H = fftw3f.create_aligned_array(FTshape, 'complex64')
        self.Ht = fftw3f.create_aligned_array(FTshape, 'complex64')
        #self.f = zeros(self.shape, 'f4')
        #self.res = zeros(self.shape, 'f4')
        #self.S = zeros((size(self.f), 3), 'f4')

        self._F = fftw3f.create_aligned_array(FTshape, 'complex64')
        self._r = fftw3f.create_aligned_array(self.shape, 'f4')
        #S0 = self.S[:,0]

        #create plans & calculate OTF and conjugate transformed OTF
        fftw3f.Plan(self.g, self.H, 'forward')()
        fftw3f.Plan(self.g2, self.Ht, 'forward')()

        self.Ht /= g.size;
        self.H /= g.size;
        
        self.H2 = self.Ht*self.H

        #calculate plans for other ffts
        self._plan_r_F = fftw3f.Plan(self._r, self._F, 'forward', flags = FFTWFLAGS, nthreads=NTHREADS)
        self._plan_F_r = fftw3f.Plan(self._F, self._r, 'backward', flags = FFTWFLAGS, nthreads=NTHREADS)
        
        fftwWisdom.save_wisdom()
        
        self.lamb = None



        
    def deconv(self, data, lamb, clip = False):
        '''This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter
       
        '''

        #test to see if we need to recalculate filter factor        
        if not lamb == self.lamb: 
            self.WF = self.Ht/(self.H2 + lamb**2)
        
        self._r[:] = data

        #F = fftn(fs)

        #d = ifftshift(ifftn(F*self.H));
        self._plan_r_F()
        self._F *= self.WF
        #self._F /= (self.H2 + lamb**2)
        self._plan_F_r()
        
        res = ifftshift(self._r)
        
        if clip:
            res = numpy.maximum(res, 0)
        
        return res