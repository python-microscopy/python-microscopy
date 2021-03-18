#!/usr/bin/python

##################
# wormlike.py
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

from scipy import *

def bareDNA(kbp, steplength=10):
    return wormlikeChain(kbp, steplength, lengthPerKbp=.34e3, persistLength=75)

def fibre30nm(kbp, steplength=10):
    return wormlikeChain(kbp, steplength, lengthPerKbp=10, persistLength=150)

class wormlikeChain:   
      
    def __init__(self, kbp, steplength=10, lengthPerKbp=10, persistLength=150):
        numsteps = round(lengthPerKbp*kbp/steplength)

        exp_costheta = (exp(-steplength/persistLength));
        theta = sqrt(2*log(1/exp_costheta))*abs(randn(numsteps, 1));
        phi = 2*pi*rand(numsteps, 1);
        #phi = 0.5*pi*randn(numsteps, 1)+pi/20;

        phi = cumsum(concatenate(([0], phi),0))
        
        xs = 1 - 2*rand()
        ys = 1 - 2*rand()
        zs = 1 - 2*rand()

        nrm = sqrt(xs**2 + ys**2 + zs**2)

        xs = xs/nrm;
        ys = ys/nrm;
        zs = zs/nrm;

        so = array([xs, ys, zs])

        xs = steplength*xs*ones(theta.shape)
        ys = steplength*ys*ones(theta.shape)
        zs = steplength*zs*ones(theta.shape)

        for i in range(2,numsteps):
            sh = cross(so, so + array([0,0,2]))
            #sh = sh./sqrt(dot(sh, sh));
            #sh = sh./sqrt(sh*sh.');
            sh = sh/dot(sh, sh.T)
            sk = cross(so, sh)
            #sk = sk./sqrt(dot(sk, sk));
            #sk = sk./sqrt(sk*sk');
            sk = sk/dot(sk, sk.T)
    
            sn = cos(theta[i])*so + sin(theta[i])*sin(phi[i])*sh + sin(theta[i])*cos(phi[i])*sk
    
            xs[i] = steplength*sn[0]
            ys[i] = steplength*sn[1]
            zs[i] = steplength*sn[2]
    
            so = sn;
            #i/numsteps


        self.xp = cumsum(concatenate(array(0), xs))
        self.yp = cumsum(concatenate(array(0), ys))
        self.zp = cumsum(concatenate(array(0), zs))

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