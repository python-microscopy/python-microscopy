#!/usr/bin/python

###############
# wormlike2.py
#
# Copyright David Baddeley, 2012
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
################



import numpy as np

def bareDNA(kbp, steplength=10):
    return WormlikeChain(kbp, steplength, lengthPerKbp=.34e3, persistLength=75.0)

def fibre30nm(kbp, steplength=10):
    return WormlikeChain(kbp, steplength, lengthPerKbp=10.0, persistLength=150.0)

def fibre10nm(kbp, steplength=10):
    return WormlikeChain(kbp, steplength, lengthPerKbp=57.0, persistLength=75.0)
    
def wiglyFibre(length, persistLength, steplength=10):
    return WormlikeChain(length, steplength, lengthPerKbp=1., persistLength=persistLength)

class WormlikeChain:         
    def __init__(self, kbp, steplength=10.0, lengthPerKbp=10.0, persistLength=150.0):
        numsteps = int(round(lengthPerKbp*kbp/steplength))

        exp_costheta = (np.exp(-steplength/persistLength))
        theta = np.sqrt(2*np.log(1/exp_costheta))*abs(np.random.randn(numsteps))
        phi = 2*np.pi*np.random.rand(numsteps)
        #phi = 0.5*pi*randn(numsteps, 1)+pi/20;

        phi = np.cumsum(np.concatenate(([0], phi),0))
        
        xs = 1 - 2*np.random.rand()
        ys = 1 - 2*np.random.rand()
        zs = 1 - 2*np.random.rand()

        nrm = np.sqrt(xs**2 + ys**2 + zs**2)

        xs = xs/nrm
        ys = ys/nrm
        zs = zs/nrm

        so = np.array([xs, ys, zs])

        xs = steplength*xs*np.ones(theta.shape)
        ys = steplength*ys*np.ones(theta.shape)
        zs = steplength*zs*np.ones(theta.shape)

        for i in range(2,numsteps):
            sh = np.cross(so, so + np.array([0,0,2]))
            sh = sh/np.dot(sh, sh.T)
            sk = np.cross(so, sh)
            sk = sk/np.dot(sk, sk.T)
    
            sn = np.cos(theta[i])*so + np.sin(theta[i])*np.sin(phi[i])*sh + np.sin(theta[i])*np.cos(phi[i])*sk
            snn = np.sqrt((sn * sn).sum())
            sn /= snn
            
            sh_n = np.cos(theta[i])*(np.cos(phi[i])*sh + np.sin(phi[i])*sk) + np.sin(theta[i])*sn
            sk_n = np.cos(theta[i])*(np.cos(phi[i])*sh + np.sin(phi[i])*sk) + np.sin(theta[i])*sn
            
            xs[i] = steplength*sn[0]
            ys[i] = steplength*sn[1]
            zs[i] = steplength*sn[2]
    
            so = sn
            #i/numsteps


        self.xp = np.cumsum(np.concatenate(([0], xs),0))
        self.yp = np.cumsum(np.concatenate(([0], ys),0))
        self.zp = np.cumsum(np.concatenate(([0], zs),0))


def line_sdf(p, verts):
    ''' 3D distance between a point and a line consisting of several segments
    
    Parameters
    ----------
    p : np.array
        (3,N) array of points to measure distance to
    verts : np.array
        (3,M)  array of points representing the vertices of the line segments
    '''
    va = verts[:, :-1] # start of each segment
    vb = verts[:, 1:] #end of each segment

    if p.ndim == 1:
        # allow us to pass in a single point
        p = np.atleast_2d(p).T

    pa, ba = p[:, :, None] - va[:,None,:], (vb - va)[:,None,:]
    # project the point onto the line
    h = np.clip((pa * ba).sum(0) / (ba * ba).sum(0), 0, 1) 
    #print(h.shape)
    d = np.sqrt(((pa - ba * h) ** 2).sum(0))
    #print(d.shape)

    return d.min(axis=1)

def dist_along_vector(p, n, verts):
    ''' 3D distance between a point and a line consisting of several segments, taken along a given direction, n
    
    Parameters
    ----------
    p : np.array
        (3,N) array of points to measure distance to
    verts : np.array
        (3,M)  array of points representing the vertices of the line segments
    '''
    va = verts[:, :-1] # start of each segment
    vb = verts[:, 1:] #end of each segment

    if p.ndim == 1:
        # allow us to pass in a single point
        p = np.atleast_2d(p).T
        n = np.atleast_2d(n).T

    pa, ba = p[:, :, None] - va[:,None,:], (vb - va)[:,None,:]
    # project the point onto the line
    h = np.clip((pa * ba).sum(0) / (ba * ba).sum(0), 0, 1) 
    #print(h.shape)
    v =(pa - ba * h)
    d = (v * v).sum(0)/(n*n).sum(0)
    #print(d.shape)

    return d.min(axis=1)

class SelfAvoidingWormlikeChain:
    #FIXME - make this actually self-avoiding - it's currently just a placeholder         
    def __init__(self, kbp, steplength=10.0, lengthPerKbp=10.0, persistLength=150.0):
        numsteps = int(round(lengthPerKbp*kbp/steplength))

        exp_costheta = (np.exp(-steplength/persistLength))
        theta = np.sqrt(2*np.log(1/exp_costheta))*abs(np.random.randn(numsteps))
        phi = 2*np.pi*np.random.rand(numsteps)
        #phi = 0.5*pi*randn(numsteps, 1)+pi/20;

        phi = np.cumsum(np.concatenate(([0], phi),0))
        
        xs = 1 - 2*np.random.rand()
        ys = 1 - 2*np.random.rand()
        zs = 1 - 2*np.random.rand()

        nrm = np.sqrt(xs**2 + ys**2 + zs**2)

        xs = xs/nrm
        ys = ys/nrm
        zs = zs/nrm

        so = np.array([xs, ys, zs])

        xs = steplength*xs*np.ones(theta.shape)
        ys = steplength*ys*np.ones(theta.shape)
        zs = steplength*zs*np.ones(theta.shape)

        for i in range(2,numsteps):
            sh = np.cross(so, so + np.array([0,0,2]))
            sh = sh/np.dot(sh, sh.T)
            sk = np.cross(so, sh)
            sk = sk/np.dot(sk, sk.T)
    
            sn = np.cos(theta[i])*so + np.sin(theta[i])*np.sin(phi[i])*sh + np.sin(theta[i])*np.cos(phi[i])*sk
            snn = np.sqrt((sn * sn).sum())
            sn /= snn
            
            sh_n = np.cos(theta[i])*(np.cos(phi[i])*sh + np.sin(phi[i])*sk) + np.sin(theta[i])*sn
            sk_n = np.cos(theta[i])*(np.cos(phi[i])*sh + np.sin(phi[i])*sk) + np.sin(theta[i])*sn
            
            xs[i] = steplength*sn[0]
            ys[i] = steplength*sn[1]
            zs[i] = steplength*sn[2]
    
            so = sn
            #i/numsteps


        self.xp = np.cumsum(np.concatenate(([0], xs),0))
        self.yp = np.cumsum(np.concatenate(([0], ys),0))
        self.zp = np.cumsum(np.concatenate(([0], zs),0))