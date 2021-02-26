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

from scipy import *

def bareDNA(kbp, steplength=10):
    return wormlikeChain(kbp, steplength, lengthPerKbp=.34e3, persistLength=75.0)

def fibre30nm(kbp, steplength=10):
    return wormlikeChain(kbp, steplength, lengthPerKbp=10.0, persistLength=150.0)

def fibre10nm(kbp, steplength=10):
    return wormlikeChain(kbp, steplength, lengthPerKbp=57.0, persistLength=75.0)
    
def wiglyFibre(length, persistLength, steplength=10):
    return wormlikeChain(length, steplength, lengthPerKbp=1., persistLength=persistLength)


def _E(cos_theta, E0=1):
    """ Bending energy as a function of angle - simplified
    
    assuming energy comes from Hookian compression and stretch of an elastic beam , the energy is proportional to kx^2
    ie sin(theta)^2
    """
    return 1 - cos_theta**2

def _E_to_costheta(E):
    return np.sqrt(1 - E)

def _prob_boltz(cos_theta, k):
    return np.exp(-_E(cos_theta)/k)

def _exp_cos_theta(k, ct_range=np.pi/2):
    """Expectation value of cos_theta for a given k"""
    
    ct = np.cos(np.linspace(0, ct_range, 500))
    
    pb = _prob_boltz(ct, k)
    
    return (ct*pb).sum()/pb.sum()
    
#calculate a lookup table of
_kvals=None
_e_ct_vals = None
def get_k_for_exp_costheta(cos_theta):
    global _kvals, _e_ct_vals
    
    if _kvals is None:
        _kvals = np.logspace(-4, 1, 500)
        _e_ct_vals = np.array([_exp_cos_theta(k) for k in _kvals])
        
    return np.interp(cos_theta, _e_ct_vals, _kvals)
        

class wormlikeChain:         
    def __init__(self, kbp, steplength=10.0, lengthPerKbp=10.0, persistLength=150.0):
        numsteps = int(np.round(lengthPerKbp*kbp/steplength))

        exp_costheta = (exp(-steplength/persistLength))

        E = np.random.exponential(1. / get_k_for_exp_costheta(exp_costheta), numsteps * 2)
        E = E[E <= 1][:numsteps]
        #theta = sqrt(2*log(1/exp_costheta))*abs(randn(numsteps))
        
        
        costheta = _E_to_costheta(E)

        sintheta = np.sqrt(1.0 - costheta ** 2)
        #phi = 2*pi*rand(numsteps);
        #phi = 0.5*pi*randn(numsteps, 1)+pi/20;

        #phi = cumsum(concatenate(([0], phi),0))
        
        
        
        xs = np.random.uniform(-1, 1, numsteps)
        ys = np.random.uniform(-1, 1, numsteps)
        zs = np.random.uniform(-1, 1, numsteps)

        nrm = sqrt(xs*xs + ys*ys + zs*zs)

        xs = xs/nrm
        ys = ys/nrm
        zs = zs/nrm
        
        #print costheta

        steps = np.array([xs, ys, zs])

        for i in range(2,numsteps):
            a = steps[:, i - 1]
            b = steps[:, i]
    
            bl = np.dot(a, b)
            bd = (b - a * bl) / np.sqrt(1 - bl * bl)
            sn = a * costheta[i] + bd * sintheta[i]
    
            steps[:, i] = sn
            
    
        xs, ys, zs = steplength*steps

        self.xp = cumsum(concatenate(([0], xs),0))
        self.yp = cumsum(concatenate(([0], ys),0))
        self.zp = cumsum(concatenate(([0], zs),0))

class wormlikeChain2D:
    def __init__(self, kbp, steplength=10.0, lengthPerKbp=10.0, persistLength=150.0):
        numsteps = int(np.round(lengthPerKbp * kbp / steplength))

        exp_costheta = (exp(-steplength/persistLength))
        #theta = sqrt(2*log(1/exp_costheta))*abs(randn(numsteps))

        #costheta = np.random.lognormal(exp_costheta, exp_costheta, numsteps)
        
        #costheta = exp_costheta*np.abs(np.random.normal(size=numsteps))/np.sqrt(2*np.pi)

        E = np.random.exponential(1. / get_k_for_exp_costheta(exp_costheta), numsteps*2)
        E = E[E <=1][:numsteps]
        #theta = sqrt(2*log(1/exp_costheta))*abs(randn(numsteps))
        costheta = _E_to_costheta(E)

        #theta = np.sqrt(2 * np.log(1 / exp_costheta)) * np.abs(randn(numsteps))
        #costheta = np.cos(theta)
        
        sintheta = np.sqrt(1.0 - costheta**2)
        #phi = 2*pi*rand(numsteps);
        #phi = 0.5*pi*randn(numsteps, 1)+pi/20;

        #phi = cumsum(concatenate(([0], phi),0))

        #print(exp_costheta, costheta.mean())

        xs = np.random.uniform(-1, 1, numsteps)
        ys = np.random.uniform(-1, 1, numsteps)
        #zs = np.random.uniform(-1, 1, numsteps)

        nrm = sqrt(xs * xs + ys * ys)# + zs * zs)

        xs = xs / nrm
        ys = ys / nrm
        #zs = zs / nrm

        #print costheta

        steps = np.array([xs, ys])

        for i in range(2, numsteps):
            a = steps[:, i-1]
            b = steps[:, i]
            
            bl = np.dot(a, b)
            bd = (b - a*bl)/np.sqrt(1 - bl*bl)
            sn = a*costheta[i] + bd*sintheta[i]

            steps[:, i] = sn

        xs, ys = steplength * steps

        self.xp = cumsum(concatenate(([0], xs), 0))
        self.yp = cumsum(concatenate(([0], ys), 0))
        self.zp = cumsum(concatenate(([0], 0*xs), 0))
        #plot3(xp, yp, zp)
        #grid
        #daspect([1 1 1])

        #if (length(xp) > 3)
        #[K, V] = convhulln([xp, yp, zp]);

        #V = V/1e9;



#xr = xp;
#yr = yp;
#zr = zp;

#end_d = sqrt((xp(1) - xp(length(xp))).^2 + (yp(1) - yp(length(xp))).^2 + (zp(1) - zp(length(xp))).^2)