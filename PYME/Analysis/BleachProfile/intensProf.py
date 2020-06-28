#!/usr/bin/python

##################
# intensProf.py
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

#from scipy import *
#from PYME.IO.FileUtils.read_kdf import *
import wx
#from pylab import *
import numpy as np
# import pylab
import matplotlib.pyplot as plt

from . import deMod

def calcProf(fn_list):
    sm = np.zeros(len(fn_list), 'f')

    inWxApp = not (wx.GetApp() is None)

    if inWxApp:
        pb = wx.ProgressDialog('Analysing data ... please wait ...', '',maximum = len(fn_list) - 1,style=wx.PD_AUTO_HIDE|wx.PD_APP_MODAL|wx.PD_REMAINING_TIME)
        pb.Show()

    for i in range(len(fn_list)): #loop over slices
        if inWxApp:
            pb.Update(i, 'Processing slice %d of %d' % (i,len(fn_list)))
        else:
            print((fn_list[i]))

        try:
            ds_a = ReadKdfData(fn_list[i])[:,:,0]
            sm[i] = ds_a.sum()
        except e:
            pass
        finally:
            pass

    return sm


def calcTs(prof, thresh):
    lOn = (prof > thresh).astype('f')
    dlOn = np.diff(lOn)

    T = np.arange(len(dlOn))
    Ton = T[dlOn > 0.5]
    
    Toff = T[dlOn < -0.5]
    if len(Ton) > len(Toff):
        Ton = Ton[1:] #remove the initial turn on of laser - we're interested in subsequent cycles

    Ibefore = prof[Toff - 1]
    Iafter = prof[Ton+2]
    Iback = prof[Toff + 2]

    return (lOn, Ton, Toff, Ibefore, Iafter, Iback)

def eMod(p, t):
    A, tau = p
    return A*(1 - np.exp(-t/tau))

def eMod5(p, t):
    A, t0, tau = p
    return A*(1 - np.exp(-(t-t0)/tau))

def eMod2(p, t):
    A, tau , b = p
    return A*(1 - b*np.exp(-t/tau))

def eMod3(p, t):
    A, tau , m = p
    return A*(1 - np.exp(-t/tau)) + m*t

def eMod4(p, t):
    A, tau1, tau2, r = p
    return A*r*(1 - np.exp(-t/tau1)) + A*(1-r)*(1 - np.exp(-t/tau2)) 
    

def doTraceDisp(prof, lOn, dt):
    T = np.arange(len(prof))*dt

    plt.clf()
    a1 = plt.axes([.1, .3,.8,.6])
    a1.plot(T, prof)
    a1.grid()
    a1.set_ylabel('Fluorescence Intensity [a.u.]')
    

    a2 = plt.axes([.1, .1,.8,.1], sharex=a1)
    a2.plot(T, lOn)
    a2.set_ylim(-.1, 1.1)
    a2.set_xlabel('Time [s]')
    a2.set_ylabel('Illumination')

    a1.set_xlim(0, T.max())
    plt.show()

    return (a1, a2)



def lNofcn(t, t0, No0, Nd0, aod, aob, ado):
    """Function to calculate number of fluorophores in on state when laser is on."""
    tn = t - t0
    
    beta = np.sqrt(aod**2 + 2*(aob + ado)*aod +aob**2 - 2*ado*aob + ado**2)

    return np.exp(-(aod + aob + ado)*tn/2)*((2*(No0 + Nd0)*ado - No0*(aod+aob + ado))*np.sinh(beta*tn/2)/beta + No0*np.cosh(beta*tn/2))


def dNofcn(t, t0, No0, Nd0, ado):
    """Function to calculate number of fluorophores in on state when laser is off."""
    tn = t - t0

    return No0 + Nd0 - Nd0*np.exp(-ado*tn)

def lNdfcn(t, t0, No0, Nd0, aod, aob, ado):
    """Function to calculate number of fluorophores in dark state when laser is on."""
    tn = t - t0
    
    beta = np.sqrt(aod**2 + 2*(aob + ado)*aod + aob**2 - 2*ado*aob + ado**2)

    return np.exp(-(aod + aob + ado)*tn/2)*((2*((No0 + Nd0)*aod +Nd0*aob)- Nd0*(aod+aob + ado))*np.sinh(beta*tn/2)/beta + Nd0*np.cosh(beta*tn/2))


def dNdfcn(t, t0, No0, Nd0, ado):
    """Function to calculate number of fluorophores in dark state when laser is off."""
    tn = t - t0

    return Nd0*np.exp(-ado*tn)


def genTraces(Ton, Toff, No0, Nd0, aod,aob, ado, dt, len_trace):
    res = np.zeros(len_trace)

    No = No0
    Nd = Nd0

    #print No, Nd
    #print aod, ado, aob

    for i in range(len(Ton) - 1):
        #calculate what happened when the laser was off
        del_t = (Ton[i] - Toff[i])*dt
        #print No, Nd
        #print del_t
        No_ = dNofcn(del_t, 0.0, No, Nd, ado)
        Nd_ = dNdfcn(del_t, 0.0, No, Nd, ado)
        #print No_, Nd_
        No = No_
        Nd = Nd_

        #print No, Nd

        #now see what happens to intensity while laser is on
        res[Ton[i]:Toff[i+1]] = lNofcn(np.arange(Ton[i], Toff[i+1])*dt, Ton[i]*dt, No, Nd, aod, aob, ado)

        #figure out what our occupancies should be at the end of the on time
        No_ = lNofcn((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado)
        Nd_ = lNdfcn((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado)
        #print No, Nd

        No = No_
        Nd = Nd_

        #print No, Nd

    return res


def traceModel(p, Ton, Toff, dt, len_trace):
    """Trace model for fitting"""
    No0, Nd0, aod,aob, ado, bg = p
    res = genTraces(Ton, Toff, No0, Nd0, aod,aob, ado, dt, len_trace)

    res = (res[Toff[0]:Toff[-1]] + bg).ravel() #just return the bit we know about
    #print res.shape
    return res



def traceModelDE(p, Ton, Toff, t, lOn):
    No0, Nd0, aod,aob, ado, bg, sc = p

    a = scipy.integrate.odeint(deMod.flFsq, [No0, Nd0], t, ((aob, aod, ado),lOn), hmax = 0.007)

    res = sc*a[:,0]*lOn

    res = (res[Toff[0]:Toff[-1]] + bg).ravel() #just return the bit we know about
    #print res.shape
    return res



def lNofcnDE(t, t0, No0, Nd0, aod, aob, ado, pow):
    """Function to calculate number of fluorophores in on state when laser is on."""
    tn = np.concatenate(((0.0,), (t - t0).reshape(-1)))
    
   
    #print tn
    #print No0, Nd0
    a = scipy.integrate.odeint(deMod.flFpow, [No0, Nd0], tn, ((aob, aod, ado,1, pow),))

    return a[1:,0]




def lNdfcnDE(t, t0, No0, Nd0, aod, aob, ado, pow):
    """Function to calculate number of fluorophores in dark state when laser is on."""
    #tn = t - t0
    tn = concatenate(((0.0,),(t - t0).reshape(-1) ))
    
    a = integrate.odeint(deMod.flFpow, [No0, Nd0], tn, ((aob, aod, ado,1, pow),))

    return a[1:,1]





def genTracesDE(Ton, Toff, No0, Nd0, aod,aob, ado, pow, dt, len_trace):
    res = np.zeros(len_trace)

    No = No0
    Nd = Nd0

    #print No, Nd
    #print aod, ado, aob

    for i in range(len(Ton) - 1):
        #calculate what happened when the laser was off
        del_t = (Ton[i] - Toff[i])*dt
        #print No, Nd
        #print del_t
        No_ = dNofcn(del_t, 0.0, No, Nd, ado)
        Nd_ = dNdfcn(del_t, 0.0, No, Nd, ado)
        #print No_, Nd_
        No = No_
        Nd = Nd_

        #print No, Nd
        #print i
        #print Ton[i], Toff[i+1]

        #now see what happens to intensity while laser is on
        #resSeg = lNofcnDE(np.arange(0, Toff[i+1] - Ton[i])*dt, 0*dt, No, Nd, aod, aob, ado)
        #print resSeg
        res[Ton[i]:Toff[i+1]] = lNofcnDE(np.arange(Ton[i], Toff[i+1])*dt, Ton[i]*dt, No, Nd, aod, aob, ado, pow)

        #figure out what our occupancies should be at the end of the on time
        No_ = lNofcnDE((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado, pow)
        Nd_ = lNdfcnDE((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado, pow)
        #print No, Nd

        No = No_[0]
        Nd = Nd_[0]

        #print No, Nd

    return res

    

def traceModelDE1(p, Ton, Toff, dt, len_trace, sc):
    """Trace model for fitting"""
    No0, Nd0, aod,aob, ado, pow, bg = p
    res = genTracesDE(Ton, Toff, No0, Nd0, aod,aob, ado, pow, dt, len_trace)

    #print res.shape
    res = (sc*res[Toff[0]:Toff[-1]] + bg).ravel() #just return the bit we know about
    #print res.shape
    return res

def mod1pk(p, t, pow=2, sc=1, bg=0, No0 = 1,Nd0 = 0):
    #No0 = 1
    #Nd0 = 0
    aod,aob, ado, pow, t0 = p
    res = lNofcnDE(t, t0, No0, Nd0**2, aod, aob, ado, pow)

    return (sc*res + bg).ravel()


def pkmod_onestate(p, t):
    No0 = 1
    Nd0 = 0
    t0,aod, aob, ado, sc = p 
    return sc*lNofcn(t, t0, No0, Nd0, aod, aob, ado)

def decmod_onestate(p, t):
    No0 = 1
    Nd0 = 0
    t0,aod, aob, ado, sc = p
    return sc*lNofcn(t, t0, No0, Nd0, aod, aob, ado)


def recolor( obj, col ):
    try: obj.set_color( col )
    except: pass
    try: obj.set_facecolor( col )
    except: pass
    try: obj.set_edgecolor( col )
    except: pass
    try:
        ch = obj.get_children()
        for c in ch:
            recolor( c, col )
    except: pass 


def doTraceDispdl2(prof, lOn, dt, Ton, Toff, Ibefore, Iafter, r, ipm, dt_ipm, dat, slx, sly, slt_init, slt_ss):
    T = np.arange(len(prof))*dt

    clf()
    a1 = axes([.1, .1,.5,.35])
    #a1.plot(T, prof, 'k')

    for i in range(len(Ton)/2 + 1):
        rt = dt*(Ton[i] - Toff[i])
        a1.plot(dt*(np.arange(Toff[i+1] - Ton[i] + 400) - 200) + 4*i + 0.0*rt,prof[(Ton[i] - 200):(Toff[i+1] + 200)] + 2*i + .00*rt, 'k')
        a1.text(4*i + 0.0*rt + dt*(Toff[i+1] - Ton[i] + .3e3),  2*i + 0.0*rt, '%d s' % (round(rt/10)*10), va='center', fontsize=10)
    #a1.grid()
    a1.set_ylabel('Fluorescence Intensity')
    a1.set_xlabel('Time [s]').set_x(.22)
    a1.set_frame_on(False)
    a1.set_xticks([0, 10])
    a1.set_yticks([])
    a1.plot([0, 30],[0,15], 'k:')
    a1.set_ylim(-2, 30)
    #a1.set_xlim(-1, 40)


    #a2 = axes([.1, .1,.5,.1], sharex=a1)
    #a2.plot(T, lOn,'k')
    #a2.set_ylim(-.1, 1.1)
    
    #a2.set_ylabel('Illumination')
    #a2.set_axis_off()

    #a1.set_xlim(0, T.max())

    #il = a2.text(-290, -0.35, 'Illumination')
    #il.set_rotation('vertical')

    a3 = axes([.1, .55, .5,.35])
    #midPeak = prof[(Ton[6] - 400):(Toff[7] + 400)]
    a3.plot((np.arange(len(ipm[:1000]))-85)*dt_ipm, ipm[:1000],'k')

    a3.plot((array(slt_init) -85)*dt_ipm, [-1, -1], 'k',lw=2)
    a3.text((slt_init[1] - 85)*dt_ipm + 0.15, -1,'I', va='center') 

    a3.plot((array(slt_ss) - 85)*dt_ipm, [-1, -1], 'k',lw=2)
    a3.text((slt_ss[1] - 85)*dt_ipm + 0.15, -1,'II', va='center') 
    
    a3.set_xlabel('Time [s]')
    a3.set_ylabel('Fluorescence Intensity [a.u.]')
    a3.set_xlim(-1,10)
    
    a3_2 = twiny(a3)
    #midPeak_z = prof[(Ton[6] - 50):(Ton[6] + 200)]
    a3_2.plot((np.arange(len(ipm[85:120])))*dt_ipm, ipm[85:120],':.', color=[.5,.5,.5])

    a3_2.plot((array(slt_init) -85)*dt_ipm, [24, 24], color=[.5,.5,.5] , lw=2)
    a3_2.text((slt_init[1] - 85)*dt_ipm + 0.01, 24,'I', va='center',color=[.5,.5,.5])
    #a3_2.plot((array(slt_ss) - 85)*dt_ipm, [24, 24],color=[.5,.5,.5] , lw=2)
    a3_2.set_xlim(-.03, .3)
    #a3_2.set_xlabel('Time [s]')

    recolor(a3_2.xaxis, [.5,.5,.5])

    a3.set_ylim(-2, 25)
    

    #a4 = axes([.7, .1, .25, .35])
    #a4.plot(((Ton - Toff)*dt)[:7], (Iafter - Ibefore)[:7]/r[0][0], 'xk')
    #a4.plot(((Ton - Toff)*dt)[7:], (Iafter - Ibefore)[7:]/r[0][0], '+k')
    #a4.plot(np.arange(1200), eMod(r[0], np.arange(1200))/r[0][0], 'k')
    #a4.set_xlabel('Time [s]')
    #a4.set_ylabel('Normalised fluorescence recovery')

    a4 = axes([.65, .1, .14, .8])
    d_c = dat[slt_init[0]:slt_init[1], slx[0]:(slx[1] + 1), sly[0]:sly[1]]
    d_c[:,slx[1] - slx[0], :] = d_c.max()
    
    a4.imshow(d_c.reshape(-1,sly[1] - sly[0] ), interpolation='nearest', cmap=cm.gray)
    a4.set_axis_off()
    a4.text(25, -7, 'I', ha='center', va='center')

    a5 = axes([.8, .1, .14, .8])
    d_c2 = dat[slt_ss[0]:slt_ss[1], slx[0]:(slx[1] + 1), sly[0]:sly[1]]
    d_c2[:,slx[1] - slx[0], :] = d_c2.max()

    d_c2[0, 3:5, (50-3 - 14):(50-3)] = d_c2.max()

    a5.imshow(d_c2.reshape(-1,sly[1] - sly[0]) , interpolation='nearest', cmap=cm.gray)
    
    #a5.plot((50 - 4) - array([2.86, 0]), [2,2] , 'w', lw=2)

    a5.set_axis_off()
    a5.text(25, -7, 'II', ha='center', va='center')

    
    a6 = axes([.15, .32, .2, .1])
    a6.plot(prof[(Toff[0] - 1e3):(Toff[1])], color=[0.5,0.5,0.5])
    a6.set_frame_on(False)
    a6.set_axis_off()
    a6.set_ylim(-.1, 7)

    a6.text((Ton[0] -Toff[0])/2 + 1e3, 2, '$\Delta t$', ha='center', va='center')
    #a6.arrow(((Ton[0] -Toff[0])/4 + 1e3),2,-(Ton[0] -Toff[0])/4,0) 
    #a6.set_xlim(0, 5e4)


    f = gcf()
    f.text(0.05, .89, 'a', fontsize=14, fontweight='bold')
    f.text(0.05, .44, 'c', fontsize=14, fontweight='bold')
    f.text(0.63, .89, 'b', fontsize=14, fontweight='bold')

    show()
    return (a1, a3, a3_2)


def doTraceDispdl(prof, lOn, dt, Ton, Toff, Ibefore, Iafter, r):
    T = np.arange(len(prof))*dt

    clf()
    a1 = axes([.1, .3,.5,.6])
    a1.plot(T, prof, 'k')
    a1.grid()
    a1.set_ylabel('Fluorescence Intensity [a.u.]')
    a1.set_xlabel('Time [s]')

    a2 = axes([.1, .1,.5,.1], sharex=a1)
    a2.plot(T, lOn,'k')
    a2.set_ylim(-.1, 1.1)
    
    #a2.set_ylabel('Illumination')
    a2.set_axis_off()

    a1.set_xlim(0, T.max())

    il = a2.text(-290, -0.35, 'Illumination')
    il.set_rotation('vertical')

    a3 = axes([.7, .55, .25,.35])
    midPeak = prof[(Ton[6] - 400):(Toff[7] + 400)]
    a3.plot((np.arange(len(midPeak)) - 400)*dt, midPeak,'k')
    a3.set_xlabel('Time [s]')
    a3.set_ylabel('Fluorescence Intensity [a.u.]')
    a3.set_xlim(-2,12)

    a4 = axes([.7, .1, .25, .35])
    a4.plot(((Ton - Toff)*dt)[:7], (Iafter - Ibefore
)[:7]/r[0][0], 'xk')
    a4.plot(((Ton - Toff)*dt)[7:], (Iafter - Ibefore
)[7:]/r[0][0], '+k')
    a4.plot(np.arange(1200), eMod2(r[0], np.arange(1200))/r[0][0], 'k')
    a4.set_xlabel('Time [s]')
    a4.set_ylabel('Normalised fluorescence recovery')


    f = gcf()
    f.text(0.05, .89, 'a', fontsize=14, fontweight='bold')
    f.text(0.63, .89, 'b', fontsize=14, fontweight='bold')
    f.text(0.63, .44, 'c', fontsize=14, fontweight='bold')

    show()
    return (a1, a2, a3, a4, il)

def eth_mod2(p, t):
    A, b, aod, ado, aob, dt = p
    theta = np.arange(-pi/2, pi/2, 0.1)
    return A*array([ethet3(theta, ti,aod, ado, aob).sum() + ethet3(theta, ti+1,aod, ado, aob).sum() for ti in t+dt])*0.1/(2*pi) + b


def ethet3(theta,t, aod, ado, aob):
    return lNofcn(t, 0, 1,0,aod*cos(theta)**2, aob*cos(theta)**2, ado)


def bf_int(func, lb, ub, *args, **kwargs):
    step = 0.1
    #step = kwargs['step']
    x = np.arange(lb,ub,step)
    v = func(x[0], *args)
    for xi in x[1:]:
        v = v + func(xi, *args)
    return v*step

def eth_mod3(p, t):                            
    A, b, aod, ado, aob, dt = p;
    return A*(bf_int(ethet3, -pi/2, pi/2, t + dt,aod, ado, aob) + bf_int(ethet3, -pi/2, pi/2, t+1 + dt,aod, ado, aob))/(2*pi) + b


def gt(theta, ot, offt, No, Nd, aod, aob, ado, dt, l):
    return genTraces(ot,offt, No, Nd, aod*cos(theta)**2, aob*cos(theta)**2, ado, dt, l)


def gt_mod(p, t, ot, offt, dt):
    A, b, No, Nd, aod, aob, ado = p    
    return A*(bf_int(gt, -pi/2,pi/2, ot, offt, No, Nd, aod, aob, ado, dt, len(t)) + bf_int(gt, -pi/2,pi/2, ot+1, offt+1, No, Nd, aod, aob, ado, dt, len(t)))/(2*pi) + b
