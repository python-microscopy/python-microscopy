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
from . import richardsonLucyMV

class MotionCompensatingRichardsonLucy(richardsonLucyMV.MultiviewRichardsonLucy):
    """Deconvolution class, implementing a variant of the Richardson-Lucy algorithm.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    

    see dec_conv for an implementation of conventional image deconvolution with a
    measured, spatially invariant PSF
    """
    
    def pd(self, g, velx, vely, tvals):
        """ remap image according to deformation field """
        g_ = g.reshape(self.dataShape[0], self.dataShape[1], -1)
        velx_ = velx.reshape(self.dataShape)
        vely_ = vely.reshape(self.dataShape)
        #print self.xv.shape, velx_.shape, tvals.shape

        #xv = np.arange(len(g))
        x_new = self.xv[:,:,None] + velx_[:, :, None] * tvals[None, None, :]
        y_new = self.yv[:,:,None] + vely_[:, :, None] * tvals[None, None, :]
        #coords = np.array([(self.xv[:, None] + shifts).T])
        coords = np.concatenate([x_new[None, :,:,:], y_new[None, :,:,:], 0*y_new[None, :,:,:]], 0)
        #print coords.shape, g_.shape, coords.T.shape
        return ndimage.map_coordinates(g_, coords, mode='wrap').ravel()
        
    def dlHd(self,data, g, velx, tvals):
        pred = self.pd(g, velx, tvals)
        
        ddx = np.array([np.gradient(r) for r in pred])
        
        #imshow(ddx)
        #imshow(data/(pred + .01))
        #imshow((data/(pred + .01) - 1)*(tvals[:, None]*ddx))
        
        return ((data/(pred + .01) - 1)*tvals[:, None]*ddx).sum(0)
        
    def dlHd2(self,data, g, velx, tvals):
        pred = self.pd(g, velx, tvals)
        
        ddx = np.array([np.gradient(r) for r in pred])
        
        #imshow(ddx)
        #imshow(data/(pred + .01))
        #imshow((data/(pred + .01) - 1)*(tvals[:, None]*ddx))
        
        return ((data/(pred + .01) - 1)*ddx).sum(0)
    
    def updateVX(self, data, g, velx, tvals, beta = -.9, nIterates = 1):
        for i in range(nIterates):
            lh0 = ndimage.gaussian_filter(self.dlHd2(data, g, velx, tvals), 50)
            lh1 = ndimage.gaussian_filter(self.dlHd2(data, g, velx + .1, tvals), 50)
            
            upd = beta*.1*lh0/(lh1-lh0 + .001)
            
            #pl.plot(lh1 - lh0)
            #plot(lh1)
            #pl.plot(upd)
            velx[:] = ndimage.gaussian_filter(velx + upd, 50)
    
    def deconv(self, views, lamb, num_iters=10, weights = 1, bg = 0, vx = 0, vy=0):
        """This is what you actually call to do the deconvolution.
        parameters are:

        data - the raw data
        lamb - the regularisation parameter
        num_iters - number of iterations (note that the convergence is fast when
                    compared to many algorithms - e.g Richardson-Lucy - and the
                    default of 10 will usually already give a reasonable result)

        alpha - PSF phase - hacked in for variable phase 4Pi deconvolution, should
                really be refactored out into the dec_4pi classes.
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

        if not np.shape(vx) == self.dataShape:
            self.vx = vx*np.ones_like(views[0]) #- 1.0#1.0
            self.vy = vy*np.ones_like(views[0])
        else:
            self.vx = vx #.ravel()
            self.vy = vy #.ravel()

        self.xv, self.yv = np.mgrid[0.:self.dataShape[0], 0.:self.dataShape[1]]

        mask = 1 - weights

        self.loopcount=0
        
        self.tVals = arange(len(views))

        while self.loopcount  < num_iters:
            self.loopcount += 1
            adjF = 0#1.0
            #adjF = 1.0
            for j in range(len(views)):

                #the residuals
                vj = views[j]
                #vj = vj/vj.sum()
                pred =  self.Afunc(self.f)
                pred = self.pd(pred, self.vx, self.vy, self.tVals[j:j+1]).squeeze()
                #pl.plot(pred)
                #pred = pred/pred.sum()
                self.res = weights*(vj/(pred +1e-1 + 0+ bg)) +  mask
                
                #adjF *= self.res
                #print vj.sum(), pred.sum()
    
                #adjustment
                adjFact = self.Ahfunc(self.res)
                adjFact = self.pd(adjFact, -self.vx, -self.vy, self.tVals[j:j+1]).squeeze()
                
                #adjF += adjFact
                adjF += adjFact
                #adjF = adjF*(.2 + .8*adjFact)
                #pl.figure()
                #pl.plot(self.res)
                #pl.plot(pred)
                #pl.plot(adjFact)
                
                #pl.figure()
                #pl.plot(adjF)
                #fnew = self.f*adjFact
                #fnew = fnew*self.f.sum()/fnew.sum()
    
            #fnew = self.f*adjF**(1./len(views))
            fnew = self.f*adjF*(1./len(views))
        
            fnew = fnew*self.f.sum()/fnew.sum()


           #set the current estimate to out new estimate
            self.f[:] = fnew
            
            #self.vx[:] = 0
            #self.updateVX(np.vstack(views), self.f, self.vx, self.tVals)
            #pl.subplot(311)
            #pl.plot(self.vx)
            #pl.subplot(312)    
            #pl.plot(fnew)
                #print(('Sum = %f' % self.f.sum()))
            
        #print 'dc3'

        return real(self.fs)


class rlbead(MotionCompensatingRichardsonLucy, dec.SpatialConvolutionMapping):
    def __init__(self, *args, **kwargs):
        MotionCompensatingRichardsonLucy.__init__(self, *args, **kwargs)

class dec_conv_slow(MotionCompensatingRichardsonLucy, dec.ClassicMappingNP):
    def __init__(self, *args, **kwargs):
        MotionCompensatingRichardsonLucy.__init__(self, *args, **kwargs)

class dec_conv(MotionCompensatingRichardsonLucy, dec.ClassicMappingFFTW):
    def __init__(self, *args, **kwargs):
        MotionCompensatingRichardsonLucy.__init__(self, *args, **kwargs)