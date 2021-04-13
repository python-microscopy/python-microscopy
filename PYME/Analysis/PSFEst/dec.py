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

import numpy as np
from scipy.linalg import *
from numpy.fft import *
from scipy import ndimage
#import weave
#import cDec

class dec:
    def subsearch(self, f0, res, fdef, Afunc, Lfunc, lam, S):
        nsrch = np.size(S,1)
        pref = Lfunc(f0-fdef)
        w0 = np.dot(pref, pref)
        c0 = np.dot(res,res)

        AS = np.zeros((np.size(res), nsrch), 'f')
        LS = np.zeros((np.size(pref), nsrch), 'f')

        for k in range(nsrch):
            AS[:,k] = np.cast['f'](Afunc(S[:,k]))
            LS[:,k] = np.cast['f'](Lfunc(S[:,k]))

        Hc = np.dot(np.transpose(AS), AS)
        Hw = np.dot(np.transpose(LS), LS)
        Gc = np.dot(np.transpose(AS), res)
        Gw = np.dot(np.transpose(-LS), pref)

        c = solve(Hc + pow(lam, 2)*Hw, Gc + pow(lam, 2)*Gw)

        cpred = c0 + np.dot(np.dot(np.transpose(c), Hc), c) - np.dot(np.transpose(c), Gc)
        wpred = w0 + np.dot(np.dot(np.transpose(c), Hw), c) - np.dot(np.transpose(c), Gw)

        fnew = f0 + np.dot(S, c)

        return (fnew, cpred, wpred)


    def deconv(self, data, lamb, num_iters=10, alpha = None):#, Afunc=self.Afunc, Ahfunc=self.Ahfunc, Lfunc=self.Lfunc, Lhfunc=self.Lhfunc):

        #lamb = 2e-2
        if (not alpha is None):
            self.alpha = alpha
            self.e1 = fftshift(np.exp(1j*self.alpha))
            self.e2 = fftshift(np.exp(2j*self.alpha))

        fdef = np.zeros(self.sliceShape, 'f').ravel()
        f = data.reshape(self.height, self.width, -1).mean(2).ravel()
        f = f/f.max()
        f = np.random.RandomState().rand(*self.sliceShape).ravel()  # replace with np.random.default_rng() on later numpy versions
#        f[self.height/2, self.width/2] = 1
#        f = f.ravel()

        S = np.zeros((np.size(f), 3), 'f')
    
        #print type(S)
        #print shape(S)

        #type(Afunc)

        nsrch = 2

        for loopcount in range(num_iters):
            pref = self.Lfunc(f - fdef);
            res = (data - self.Afunc(f));
        
            #print type(Afunc(res))
            #print shape(Afunc(res))
            
            #print pref.typecode()
            
            S[:,0] = np.cast['f'](self.Ahfunc(res))
            S[:,1] = np.cast['f'](-self.Lhfunc(pref))

            #print S

            test = 1 - abs(np.dot(S[:,0], S[:,1])/(norm(S[:,0])*norm(S[:,1])))

            print(('Test Statistic %f\n' % (test,)))

            (fnew, cpred, spred) = self.subsearch(f, res, fdef, self.Afunc, self.Lfunc, lamb, S[:, 0:nsrch])

            fnew = np.cast['f'](fnew*(fnew > 0))

            S[:,2] = np.cast['f'](fnew - f)
            nsrch = 3

            f = fnew

        return np.real(f)
        
    def sim_pic(self,data,alpha):
        self.alpha = alpha
        self.e1 = fftshift(np.exp(1j*self.alpha))
        self.e2 = fftshift(np.exp(2j*self.alpha))
        
        return self.Afunc(data)



class dec_psf(dec):
    def __init__(self,points, sliceShape):
        self.sliceShape = sliceShape
        self.height, self.width = sliceShape
        self.points = points

        self.prepare()
        
    def psf_calc(self, psf, kz, data_size):
        #x1 = arange(-floor(data_size[0]/2),(ceil(data_size[0]/2)))
        #y1 = arange(-floor(data_size[1]/2),(ceil(data_size[1]/2)))
        #z1 = arange(-floor(data_size[2]/2),(ceil(data_size[2]/2)))

        #kz = 1.4661 1.4661*2*pi/(lambda*voxelsize.z);
#        g = psf;
#
#        self.height = data_size[0]
#        self.width  = data_size[1]
#        self.depth  = data_size[2]
#
#        (x,y,z) = mgrid[-floor(self.height/2.0):(ceil(self.height/2.0)), -floor(self.width/2.0):(ceil(self.width/2.0)), -floor(self.depth/2.0):(ceil(self.depth/2.0))]
#
#        #print x
#
#        #x = single(x);
#        #y = single(y);
#        #z = single(z);
#
#        #g = padarray(psf, [height - size(psf,1), width - size(psf,2), depth - size(psf,3)], 0, 'post');
#
#
#        #g = circshift(g, floor([(height - size(pssm,1))./2, (width - size(pssm,2))./2, (depth - size(pssm,3))./2]));
#        gs = shape(g);
#
#        #print (gs[2] - self.depth)/2
#        #print (self.depth + floor((gs[2] - self.depth)/2))
#        g = g[int(floor((gs[0] - self.height)/2)):int(self.height + floor((gs[0] - self.height)/2)), int(floor((gs[1] - self.width)/2)):int(self.width + floor((gs[1] - self.width)/2)), int(floor((gs[2] - self.depth)/2)):int(self.depth + floor((gs[2] - self.depth)/2))]
#
#        #g2 = zeros(shape(g), 'f')
#        #g2[0:self.height, 0:self.width, 0:self.depth] = g
#
#        g = abs(ifftshift(ifftn(abs(fftn(g)))));
#        g = (g/sum(sum(sum(g))));
#
#        self.g = g;
#
#        #%g = circshift(g, [0, -1]);
#        self.H = cast['f'](fftn(g));
#        self.Ht = cast['f'](ifftn(g));
#
#        tk = 2*kz*z
#        #t = g*sin(tk)
#        #self.Hs = fftn(t);
#        #self.Hst = ifftn(t);
#
#        #t = g*cos(tk)
#        #self.Hc = fftn(t);
#        #self.Hct = ifftn(t);
#
#        t = g*exp(1j*tk)
#        self.He = cast['F'](fftn(t));
#        self.Het = cast['F'](ifftn(t));
#
#        tk = 2*tk
#        #t = g*sin(tk)
#        #self.Hs2 = fftn(t);
#        #self.Hs2t = ifftn(t);
#
#        #t = g*cos(tk)
#        #self.Hc2 = fftn(t);
#        #self.Hc2t = ifftn(t);
#
#        t = g*exp(1j*tk)
#        self.He2 = cast['F'](fftn(t));
#        self.He2t = cast['F'](ifftn(t));
        pass

    def prepare(self):
        kx,ky = np.mgrid[:self.sliceShape[0],:self.sliceShape[1]]#,:self.sliceShape[2]]

        self.kx = fftshift(kx - self.sliceShape[0]/2.)/self.sliceShape[0]
        self.ky = fftshift(ky - self.sliceShape[1]/2.)/self.sliceShape[1]
        #self.kz = fftshift(kz - self.sliceShape[2]/2.)/self.sliceShape[2]

        return True

#    def Lfunc(self, f):
#        fs = reshape(f, (self.height, self.width, self.depth))
#        a = -6*fs
#
#        a[:,:,0:-1] += fs[:,:,1:]
#        a[:,:,1:] += fs[:,:,0:-1]
#
#        a[:,0:-1,:] += fs[:,1:,:]
#        a[:,1:,:] += fs[:,0:-1,:]
#
#        a[0:-1,:,:] += fs[1:,:,:]
#        a[1:,:,:] += fs[0:-1,:,:]
#
#        return ravel(cast['f'](a))

    def Lfunc(self, f):
        fs = np.reshape(f, (self.height, self.width))
#        a = -4*fs
#
#        a[:,0:-1] += fs[:,1:]
#        a[:,1:] += fs[:,0:-1]
#
#        a[0:-1,:] += fs[1:,:]
#        a[1:,:] += fs[0:-1,:]

        a = ndimage.convolve(fs, np.array([[0, -1, 0], [-1, 4, -1],[0,-1,0]]))

        return np.ravel(np.cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        fs = np.reshape(f, (self.height, self.width))

        F = fftn(fs)

        d = np.zeros((self.height, self.width, len(self.points)))

        for i in range(len(self.points)):
            p = self.points[i,:]
            d[:,:,i] = p[2]*ifftn(F*np.exp(-2j*np.pi*(self.kx*p[0] + self.ky*p[1]))).real

        return np.ravel(d)

    def Ahfunc(self, f):
        fs = np.reshape(f, (self.height, self.width, len(self.points)))

        d = np.zeros((self.height, self.width))

        for i in range(len(self.points)):
            F = fftn(fs[:,:,i])
            p = self.points[i,:]
            d = d + ifftn(F*np.exp(-2j*np.pi*(self.kx*-p[0] + self.ky*-p[1]))).real
        
        return np.ravel(d/(self.points[:,2].sum()))
    
#class dec_4pi_c(dec_4pi):
#    def prepare(self):
#        return cDec.prepare(shape(self.H))
#
#    def cleanup(self):
#        return cDec.cleanup()
#
#    def Afunc(self, f):
#        return ravel(ifftshift(reshape(cDec.fw_map(cast['F'](f),cast['F'](self.alpha), cast['F'](self.H), cast['F'](self.He), cast['F'](self.He2), cast['F'](self.e1), cast['F'](self.e2)), shape(self.alpha))))
#
#    def Ahfunc(self, f):
#        #return cDec.fw_map(f,self.alpha, self.Ht, self.Het, self.He2t, self.e1, self.e2)
#        return ravel(ifftshift(reshape(cDec.fw_map(cast['F'](f),cast['F'](self.alpha), cast['F'](self.Ht), cast['F'](self.Het), cast['F'](self.He2t), cast['F'](self.e1), cast['F'](self.e2)), shape(self.alpha))))
#
#    def Lfunc(self,f):
#        return cDec.Lfunc(f, shape(self.alpha))
#
#    Lhfunc = Lfunc
  
        