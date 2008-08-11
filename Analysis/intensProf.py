from scipy import *
from PYME.FileUtils.read_kdf import *
import wx
from pylab import *

import deMod

def calcProf(fn_list):
    sm = zeros(len(fn_list), 'f')

    inWxApp = not (wx.GetApp() == None)

    if inWxApp:
        pb = wx.ProgressDialog('Analysing data ... please wait ...', '',maximum = len(fn_list) - 1,style=wx.PD_AUTO_HIDE|wx.PD_APP_MODAL|wx.PD_REMAINING_TIME)
        pb.Show()

    for i in range(len(fn_list)): #loop over slices
        if inWxApp:
            pb.Update(i, 'Processing slice %d of %d' % (i,len(fn_list)))
        else:
            print fn_list[i]

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
    dlOn = diff(lOn)

    T = arange(len(dlOn))
    Ton = T[dlOn > 0.5]
    Ton = Ton[1:] #remove the initial turn on of laser - we're interested in subsequent cycles
    Toff = T[dlOn < -0.5]

    Ibefore = prof[Toff - 1]
    Iafter = prof[Ton+2]
    Iback = prof[Toff + 2]

    return (lOn, Ton, Toff, Ibefore, Iafter, Iback)

def eMod(p, t):
    A, tau = p
    return A*(1 - exp(-t/tau))

def eMod2(p, t):
    A, tau , b = p
    return A*(1 - b*exp(-t/tau))
    

def doTraceDisp(prof, lOn, dt):
    T = arange(len(prof))*dt

    clf()
    a1 = axes([.1, .3,.8,.6])
    a1.plot(T, prof)
    a1.grid()
    a1.set_ylabel('Fluorescence Intensity [a.u.]')
    

    a2 = axes([.1, .1,.8,.1], sharex=a1)
    a2.plot(T, lOn)
    a2.set_ylim(-.1, 1.1)
    a2.set_xlabel('Time [s]')
    a2.set_ylabel('Illumination')

    a1.set_xlim(0, T.max())
    show()

    return (a1, a2)



def lNofcn(t, t0, No0, Nd0, aod, aob, ado):
    '''Function to calculate number of fluorophores in on state when laser is on.'''
    tn = t - t0
    
    beta = sqrt(aod**2 + 2*(aob + ado)*aod +aob**2 - 2*ado*aob + ado**2)

    return exp(-(aod + aob + ado)*tn/2)*((2*(No0 + Nd0)*ado - No0*(aod+aob + ado))*sinh(beta*tn/2)/beta + No0*cosh(beta*tn/2))


def dNofcn(t, t0, No0, Nd0, ado):
    '''Function to calculate number of fluorophores in on state when laser is off.'''
    tn = t - t0

    return No0 + Nd0 - Nd0*exp(-ado*tn)

def lNdfcn(t, t0, No0, Nd0, aod, aob, ado):
    '''Function to calculate number of fluorophores in dark state when laser is on.'''
    tn = t - t0
    
    beta = sqrt(aod**2 + 2*(aob + ado)*aod + aob**2 - 2*ado*aob + ado**2)

    return exp(-(aod + aob + ado)*tn/2)*((2*((No0 + Nd0)*aod +Nd0*aob)- Nd0*(aod+aob + ado))*sinh(beta*tn/2)/beta + Nd0*cosh(beta*tn/2))


def dNdfcn(t, t0, No0, Nd0, ado):
    '''Function to calculate number of fluorophores in dark state when laser is off.'''
    tn = t - t0

    return Nd0*exp(-ado*tn)


def genTraces(Ton, Toff, No0, Nd0, aod,aob, ado, dt, len_trace):
    res = zeros(len_trace)

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
        res[Ton[i]:Toff[i+1]] = lNofcn(arange(Ton[i], Toff[i+1])*dt, Ton[i]*dt, No, Nd, aod, aob, ado)

        #figure out what our occupancies should be at the end of the on time
        No_ = lNofcn((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado)
        Nd_ = lNdfcn((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado)
        #print No, Nd

        No = No_
        Nd = Nd_

        #print No, Nd

    return res


def traceModel(p, Ton, Toff, dt, len_trace):
    '''Trace model for fitting'''
    No0, Nd0, aod,aob, ado, bg = p
    res = genTraces(Ton, Toff, No0, Nd0, aod,aob, ado, dt, len_trace)

    res = (res[Toff[0]:Toff[-1]] + bg).ravel() #just return the bit we know about
    #print res.shape
    return res



def traceModelDE(p, Ton, Toff, t, lOn):
    No0, Nd0, aod,aob, ado, bg, sc = p

    a = integrate.odeint(deMod.flFsq, [No0, Nd0], t, ((aob, aod, ado),lOn), hmax = 0.007)

    res = sc*a[:,0]*lOn

    res = (res[Toff[0]:Toff[-1]] + bg).ravel() #just return the bit we know about
    #print res.shape
    return res



def lNofcnDE(t, t0, No0, Nd0, aod, aob, ado, pow):
    '''Function to calculate number of fluorophores in on state when laser is on.'''
    tn = concatenate(((0.0,), (t - t0).reshape(-1)))
    
   
    #print tn
    #print No0, Nd0
    a = integrate.odeint(deMod.flFpow, [No0, Nd0], tn, ((aob, aod, ado,1, pow),))

    return a[1:,0]




def lNdfcnDE(t, t0, No0, Nd0, aod, aob, ado, pow):
    '''Function to calculate number of fluorophores in dark state when laser is on.'''
    #tn = t - t0
    tn = concatenate(((0.0,),(t - t0).reshape(-1) ))
    
    a = integrate.odeint(deMod.flFpow, [No0, Nd0], tn, ((aob, aod, ado,1, pow),))

    return a[1:,1]





def genTracesDE(Ton, Toff, No0, Nd0, aod,aob, ado, pow, dt, len_trace):
    res = zeros(len_trace)

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
        #resSeg = lNofcnDE(arange(0, Toff[i+1] - Ton[i])*dt, 0*dt, No, Nd, aod, aob, ado)
        #print resSeg
        res[Ton[i]:Toff[i+1]] = lNofcnDE(arange(Ton[i], Toff[i+1])*dt, Ton[i]*dt, No, Nd, aod, aob, ado, pow)

        #figure out what our occupancies should be at the end of the on time
        No_ = lNofcnDE((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado, pow)
        Nd_ = lNdfcnDE((Toff[i+1] - Ton[i])*dt, 0, No, Nd, aod, aob, ado, pow)
        #print No, Nd

        No = No_[0]
        Nd = Nd_[0]

        #print No, Nd

    return res

    

def traceModelDE1(p, Ton, Toff, dt, len_trace, sc):
    '''Trace model for fitting'''
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
