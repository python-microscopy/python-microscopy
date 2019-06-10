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
#from scipy import *
#from scipy.linalg import *
#from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy import ndimage
import numpy
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
import scipy
import PYME.misc.fftw_compat as fftw3f
from . import fftwWisdom

fftwWisdom.load_wisdom()

#import weave
#import cDec
#from PYME import pad
#import dec

NTHREADS = 1
FFTWFLAGS = ['measure']

def resizePSF(psf, data_size):
    if not psf.shape == data_size:
        #Expand PSF to data size by fourier domain interpolation
        print('Resizing PSF to match data size')
        g_ = fftw3f.create_aligned_array(data_size, 'complex64')
        H_ = fftw3f.create_aligned_array(data_size, 'complex64')
        
        sx, sy, sz = numpy.array(data_size).astype('f') / psf.shape
        
        #print sx, sy, sz
        
        OT = fftshift(fftn(fftshift(psf))) #don't bother with FFTW here as raw PSF is small
        
        if data_size[2] > 1:
            pr = ndimage.zoom(OT.real, [sx,sy,sz], order=1)
            pi = ndimage.zoom(OT.imag, [sx,sy,sz], order=1)
        else: #special case for 2D
            pr = ndimage.zoom(OT.real.squeeze(), [sx,sy], order=1).reshape(data_size)
            pi = ndimage.zoom(OT.imag.squeeze(), [sx,sy], order=1).reshape(data_size)
        
        H_[:] = ifftshift(pr + 1j*pi)
        fftw3f.Plan(H_, g_, 'backward')()
        #View3D(psf)
        #View3D(H_)
        #View3D(OT)
        #View3D(pr)
        
        g =  ifftshift(g_.real).clip(min=0) # negative values may cause instability
    
        print('PSF resizing complete')
    else:
        g = psf
    #View3D(psf)
    #View3D(fftshift(numpy.abs(H_)))
    #View3D(fftshift(numpy.angle(H_)))
    #View3D(g)
    
    return g/g.sum()


class dec_wiener:
    """Classical deconvolution with a stationary PSF"""
    def psf_calc(self, psf, data_size):
        """Precalculate the OTF etc..."""
        
        g = resizePSF(psf, data_size)
        

        #keep track of our data shape
        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size

        FTshape = [self.shape[0], self.shape[1], self.shape[2]/2 + 1]

        #allocate memory
       
        self.H = fftw3f.create_aligned_array(FTshape, 'complex64')
        self.Ht = fftw3f.create_aligned_array(FTshape, 'complex64')

        self._F = fftw3f.create_aligned_array(FTshape, 'complex64')
        self._r = fftw3f.create_aligned_array(self.shape, 'f4')
        
        
        
        self.g = g.astype('f4');
        self.g2 = 1.0*self.g[::-1, ::-1, ::-1]

        
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
        """This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter
       
        """

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
