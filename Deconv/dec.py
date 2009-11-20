#!/usr/bin/python

##################
# dec.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from scipy import * 
from scipy.linalg import * 
import weave
import cDec

class dec:
    def subsearch(self, f0, res, fdef, Afunc, Lfunc, lam, S):
        nsrch = size(S,1)
        pref = Lfunc(f0-fdef)
        w0 = dot(pref, pref)
        c0 = dot(res,res)

        AS = zeros((size(res), nsrch), 'f')
        LS = zeros((size(pref), nsrch), 'f')

        for k in range(nsrch):
            AS[:,k] = cast['f'](Afunc(S[:,k]))
            LS[:,k] = cast['f'](Lfunc(S[:,k]))

        Hc = dot(transpose(AS), AS)
        Hw = dot(transpose(LS), LS)
        Gc = dot(transpose(AS), res)
        Gw = dot(transpose(-LS), pref)

        c = solve(Hc + pow(lam, 2)*Hw, Gc + pow(lam, 2)*Gw)

        cpred = c0 + dot(dot(transpose(c), Hc), c) - dot(transpose(c), Gc)
        wpred = w0 + dot(dot(transpose(c), Hw), c) - dot(transpose(c), Gw)

        fnew = f0 + dot(S, c)

        return (fnew, cpred, wpred)


    def deconv(self, data, lamb, num_iters=10, alpha = None):#, Afunc=self.Afunc, Ahfunc=self.Ahfunc, Lfunc=self.Lfunc, Lhfunc=self.Lhfunc):

        #lamb = 2e-2
        if (not alpha == None):
            self.alpha = alpha
            self.e1 = fftshift(exp(1j*self.alpha))
            self.e2 = fftshift(exp(2j*self.alpha))

        fdef = zeros(shape(data), 'f')
        f = data

        S = zeros((size(f), 3), 'f')
    
        #print type(S)
        #print shape(S)

        #type(Afunc)

        nsrch = 2

        for loopcount in range(num_iters):
            pref = self.Lfunc(f - fdef);
            res = data - self.Afunc(f);
        
            #print type(Afunc(res))
            #print shape(Afunc(res))
            
            #print pref.typecode()
            
            S[:,0] = cast['f'](self.Ahfunc(res))
            S[:,1] = cast['f'](-self.Lhfunc(pref))

            #print S

            test = 1 - abs(dot(S[:,0], S[:,1])/(norm(S[:,0])*norm(S[:,1])))

            print 'Test Statistic %f\n' % (test,)

            (fnew, cpred, spred) = self.subsearch(f, res, fdef, self.Afunc, self.Lfunc, lamb, S[:, 0:nsrch])

            fnew = cast['f'](fnew*(fnew > 0))

            S[:,2] = cast['f'](fnew - f)
            nsrch = 3

            f = fnew

        return real(f)
        
    def sim_pic(self,data,alpha):
        self.alpha = alpha
        self.e1 = fftshift(exp(1j*self.alpha))
        self.e2 = fftshift(exp(2j*self.alpha))
        
        return self.Afunc(data)



class dec_4pi(dec):
    def psf_calc(self, psf, kz, data_size):
        #x1 = arange(-floor(data_size[0]/2),(ceil(data_size[0]/2)))
        #y1 = arange(-floor(data_size[1]/2),(ceil(data_size[1]/2)))
        #z1 = arange(-floor(data_size[2]/2),(ceil(data_size[2]/2)))

        #kz = 1.4661 1.4661*2*pi/(lambda*voxelsize.z);
        g = psf;

        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        (x,y,z) = mgrid[-floor(self.height/2.0):(ceil(self.height/2.0)), -floor(self.width/2.0):(ceil(self.width/2.0)), -floor(self.depth/2.0):(ceil(self.depth/2.0))]
        
        #print x
        
        #x = single(x);
        #y = single(y);
        #z = single(z);

        #g = padarray(psf, [height - size(psf,1), width - size(psf,2), depth - size(psf,3)], 0, 'post');
        
        
        #g = circshift(g, floor([(height - size(pssm,1))./2, (width - size(pssm,2))./2, (depth - size(pssm,3))./2]));
	gs = shape(g);
	
	#print (gs[2] - self.depth)/2
	#print (self.depth + floor((gs[2] - self.depth)/2))
        g = g[int(floor((gs[0] - self.height)/2)):int(self.height + floor((gs[0] - self.height)/2)), int(floor((gs[1] - self.width)/2)):int(self.width + floor((gs[1] - self.width)/2)), int(floor((gs[2] - self.depth)/2)):int(self.depth + floor((gs[2] - self.depth)/2))]
	
	#g2 = zeros(shape(g), 'f')
	#g2[0:self.height, 0:self.width, 0:self.depth] = g

        g = abs(ifftshift(ifftn(abs(fftn(g)))));
        g = (g/sum(sum(sum(g))));
	
	self.g = g;
        
        #%g = circshift(g, [0, -1]);
        self.H = cast['f'](fftn(g));
        self.Ht = cast['f'](ifftn(g));

        tk = 2*kz*z
        #t = g*sin(tk)
        #self.Hs = fftn(t);    
        #self.Hst = ifftn(t);
        
        #t = g*cos(tk)
        #self.Hc = fftn(t);
        #self.Hct = ifftn(t);
        
        t = g*exp(1j*tk)
        self.He = cast['F'](fftn(t));    
        self.Het = cast['F'](ifftn(t));

        tk = 2*tk
        #t = g*sin(tk)
        #self.Hs2 = fftn(t);
        #self.Hs2t = ifftn(t);

        #t = g*cos(tk)
        #self.Hc2 = fftn(t);
        #self.Hc2t = ifftn(t);
        
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
        #d_cos = ifftshift(ifftn(F*self.Hc));
        #d_sin = ifftshift(ifftn(F*self.Hs));
        d_e = ifftshift(ifftn(F*self.He));
        #d_cos2 = ifftshift(ifftn(F*self.Hc2));
        #d_sin2 = ifftshift(ifftn(F*self.Hs2));
        d_e2 = ifftshift(ifftn(F*self.He2));

        #d = (1.5*d_1 + 2*cos(self.alpha)*d_cos - 2*sin(self.alpha)*d_sin + 0.5*cos(2*self.alpha)*d_cos2 - 0.5*sin(2*self.alpha)*d_sin2);
        #d = (1.5*d_1 + 2*real(d_e*exp(1j*self.alpha)) + 0.5*real(d_e2*exp(2*1j*self.alpha)));
        d = (1.5*real(d_1) + 2*real(d_e*self.e1) + 0.5*real(d_e2*self.e2))
        #e1 = self.e1
        #e2 = self.e2
        
        #d = d_1.copy()
        #weave.blitz('d = (1.5*(d_1) + 2*(d_e*e1) + 0.5*(d_e2*e2))')
        
        d = real(d);
        return ravel(d)

    def Ahfunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))

        F = fftn(fs)

        d_1 = ifftshift(ifftn(F*self.Ht));
        #d_cos = ifftshift(ifftn(F*self.Hct));
        #d_sin = ifftshift(ifftn(F*self.Hst));
        d_e = ifftshift(ifftn(F*self.Het));
        #d_cos2 = ifftshift(ifftn(F*self.Hc2t));
        #d_sin2 = ifftshift(ifftn(F*self.Hs2t));
        d_e2 = ifftshift(ifftn(F*self.He2t));

        #d = (1.5*d_1 + 2*cos(self.alpha)*d_cos - 2*sin(self.alpha)*d_sin + 0.5*cos(2*self.alpha)*d_cos2 - 0.5*sin(2*self.alpha)*d_sin2);
        d = (1.5*d_1 + 2*real(d_e*exp(1j*self.alpha)) + 0.5*real(d_e2*exp(2*1j*self.alpha)));
        
        d = real(d);
        return ravel(d)

class dec_conv(dec):
    def psf_calc(self, psf, data_size):
        g = psf;

        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        (x,y,z) = mgrid[-floor(self.height/2.0):(ceil(self.height/2.0)), -floor(self.width/2.0):(ceil(self.width/2.0)), -floor(self.depth/2.0):(ceil(self.depth/2.0))]

        gs = shape(g);

        g = g[int(floor((gs[0] - self.height)/2)):int(self.height + floor((gs[0] - self.height)/2)), int(floor((gs[1] - self.width)/2)):int(self.width + floor((gs[1] - self.width)/2)), int(floor((gs[2] - self.depth)/2)):int(self.depth + floor((gs[2] - self.depth)/2))]

        #g = abs(ifftshift(ifftn(abs(fftn(g)))));
        g = (g/sum(sum(sum(g))));

        self.g = g;

        #%g = circshift(g, [0, -1]);
        self.H = cast['f'](fftn(g));
        self.Ht = cast['f'](ifftn(g));


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

        d = ifftshift(ifftn(F*self.H));

        d = real(d);
        return ravel(d)

    def Ahfunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))

        F = fftn(fs)
        d = ifftshift(ifftn(F*self.Ht));
        d = real(d);
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
  
        