#!/usr/bin/python

###############
# rawIntensity.py
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


#from pylab import *
import numpy as np
from PYME.Analysis._fithelpers import *

def dyedyemod(p, t):
    """Model for bleach decay when the rate is dominated by dye-dye interactions.
    The equation (~ 1/t) is a solution to a differential equation of the form:
        dN/dt = -k*N^2
        
    The parameters are:
        A0 - The initial intensity
        k - the rate constant
        b - a constant background
        
    Note that the 1/t curve is shifted to be 1 at t=0."""
        
    A0, k, b = p
    return A0/(t/k + 1) + b
    
def dyedyemoda(p, t):
    """Model for bleach decay when the rate is dominated by dye-dye interactions.
    The equation (~ 1/t) is a solution to a differential equation of the form:
        dN/dt = -k*N^2
        
    The parameters are:
        A0 - The initial intensity
        k - the rate constant
        b - a constant background
        
    Note that the 1/t curve is shifted to be 1 at t=0."""
        
    A0, k, b = p
    return A0**2/(t/k + 1) + b
    
def dyedyemodt(p, t):
    """Model for bleach decay when the rate is dominated by dye-dye interactions.
    The equation (~ 1/t) is a solution to a differential equation of the form:
        dN/dt = -k*N^2
        
    The parameters are:
        A0 - The initial intensity
        k - the rate constant
        b - a constant background
        
    Note that the 1/t curve is shifted to be 1 at t=0."""
        
    A0, k, b, c = p
    return A0/((t/k)**c + 1) + b

def dyedyemod2(p, t, dataStart):
    """Model for bleach decay when the rate is dominated by dye-dye interactions.
    The equation (~ 1/t) is a solution to a differential equation of the form:
        dN/dt = -k*N^2
        
    The parameters are:
        A0 - The initial intensity
        k - the rate constant
        b - a constant background
        o - an offset between the parts of the curves taken with different EM gain
        
    dataStart dictates the start of the protion with high EM gain
        
    Note that the 1/t curve is shifted to be 1 at t=0."""
        
    A0, k, b, o = p
    return A0/(t/k + 1) + b + o*(t<dataStart)
    
def dyedyemod3(p, t, dataStart):
    """Model for bleach decay when the rate is dominated by dye-dye interactions.
    The equation (~ 1/t) is a solution to a differential equation of the form:
        dN/dt = -k*N^2
        
    The parameters are:
        A0 - The initial intensity
        k - the rate constant
        b - a constant background
        o - an offset between the parts of the curves taken with different EM gain
        
    dataStart dictates the start of the protion with high EM gain
        
    Note that the 1/t curve is shifted to be 1 at t=0."""
        
    A0, k, b, sc, o = p
    return ((sc - 1)*(t<dataStart) + 1)*A0/(t/k + 1) + b + o*(t<dataStart)
    
def linMod(p, t):
    m1, m2, b, o = p
    
    return m1*t +b + (t<= 0)*(o + m2*t)
    

def processIntensityTrace(I, mdh, dt=1):
    #from pylab import *
    import matplotlib.pyplot as plt
    
    t = np.arange(len(I))*dt
    dfb, dfe = mdh['Protocol.DarkFrameRange']
    dk_emgain = I[dfb:dfe].mean()
    
    pbb, pbe = mdh['Protocol.PrebleachFrames']
    bb, be = mdh['Protocol.BleachFrames']
    
    #this is asuming that we get a couple of dark frames between the prebleach 
    #exposure and the high intensity switch on.
    dk_noem = min(I[pbe:be])
    
    #find real start of bleach curve by looking for the maximum of the bleach signal
    bb += np.argmax(I[bb:be])
    bb += 1
    
    init_bleach = mdh['Camera.TrueEMGain']*(I[bb:be] - dk_noem)
    t_init = t[bb:be]
    
    ds = mdh['Protocol.DataStartsAt']
    
    full_bleach = I[ds:] - dk_emgain
    t_full = t[ds:] + .3
    
    
    
    r0 = FitModel(linMod, [-1, -1, full_bleach[0],full_bleach[3] - init_bleach[-1]], np.hstack((init_bleach[-40:], full_bleach[5:40])),
                  np.hstack((t_init[-40:], t_full[5:40])) - t_init[-1])
    print((r0[0]))
    
    plt.figure()
    t_ = np.hstack((t_init[-40:], t_full[5:40])) - t_init[-1]
    plt.plot(t_, np.hstack((init_bleach[-40:], full_bleach[5:40])))
    plt.plot(t_, linMod(r0[0], t_))
    
    init_bleach = init_bleach - r0[0][3]#full_bleach[5] - init_bleach[-1]
    
    #figure()
    #plot(t, I)
    
    t2 = np.hstack([t_init, t_full])
    I2 = np.hstack([init_bleach, full_bleach])
    
    t3 = t2 - t2[0]
    
    #ds3 = t[ds] - t2[0]
    
    r = FitModelWeighted(dyedyemod, [I2[0], 1., I2[-1]], I2, np.sqrt(I2)/4, t3)
    print((r[0]))
    
    #r2 = FitModel(dyedyemod2, [I2[0], 1., I2[-1], 0.], I2, t3, ds3)
    #print r2[0]
    
    r3 = FitModel(dyedyemod, [init_bleach[0], .01, full_bleach[-1]], init_bleach, t_init - t_init[0])
    print((r3[0]))
    
    print([init_bleach[0], .01, full_bleach[-1]])
    
    r4 = FitModel(dyedyemod, [init_bleach[0], .1, full_bleach[-1]], full_bleach, t_full - t_init[0])
    print((r4[0]))
    
    #r5 = FitModel(dyedyemoda, [sqrt(init_bleach[0]), .01, full_bleach[-1]], full_bleach, t_full - t_init[0])
    #print r5[0]
    
    A0 = init_bleach[0] 
    print(A0)
    r6 = FitModelWeighted(dyedyemodt, [A0, dt, full_bleach[-1], 1.], I2, np.sqrt(I2)/4, t3)
    print((r6[0]))
    
    plt.figure()
    plt.plot(t_init, init_bleach)
    plt.plot(t_full, full_bleach)
    
    plt.plot(t2, dyedyemod(r[0], t3))
    #plot(t2, dyedyemod2(r2[0], t3, ds3))
    
    #plot(t2, dyedyemod(r3[0], t3))
    plt.plot(t2, dyedyemod(r4[0], t3) )
    
    plt.xlabel('Time [s]')
    plt.ylabel('Intensity')
    #plot(t_init, init_bleach + full_bleach[0] - init_bleach[-1])
    
    plt.figure()
    
    plt.loglog(t3 + dt, I2)
    plt.loglog(t3 + dt, dyedyemod(r[0], t3))
    #loglog(t3, dyedyemod2(r2[0], t3, ds3))
    
    #loglog(t3, dyedyemod(r3[0], t3))
    #loglog(t3 + dt, dyedyemod(r4[0], t3))
    #loglog(t3, dyedyemoda(r5[0], t3))
    plt.loglog(t3+ dt, dyedyemodt(r6[0], t3))
    #loglog(t3+ dt, dyedyemodt([dt,I2[-1]], t3, A0))
    
    
    plt.figtext(.3,.8, 'A0 = %3.0f, k = %3.4f, b = %3.1f' % tuple(r[0]), size=18, color='g')
    #figtext(.3,.75, 'A0 = %3.0f, k = %3.4f, b = %3.1f' % tuple(r4[0]), size=18, color='r')
    plt.figtext(.3,.75, 'A0 = %3.0f, k = %3.3f, b = %3.1f, exp = %3.3f' % tuple(r6[0]), size=18, color='r')
    #i2s = 
    #plot(t2[:-1], diff(convolve(I2, ones(20), 'same')))
    
    #return t2, I2
    
    
    
    
    
    