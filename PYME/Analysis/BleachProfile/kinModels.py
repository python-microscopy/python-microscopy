#!/usr/bin/python

###############
# kinModels.py
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
from pylab import *
from PYME.Analysis._fithelpers import *
from PYME.Analysis.DeClump import deClump
from scipy.special import erf
from PYME.Analysis.processLogger import PL
import inspect
import numpy as np
import os

colours = ['r', 'g', 'b']

USE_GUI = True

def FitModelG(*args):
    return FitModel(*args)[0]

def munge_res(model, res, **kwargs):
    #res = FitModel(model, startParams, data, *args)
    #if mse:
    r = np.hstack([res[0],] + list(kwargs.values()))

    dt = np.dtype({'names':model.paramNames + list(kwargs.keys()), 'formats':len(r)*[r.dtype.str]})
    #else:
    #    r = res[0]
    #
    #    dt = np.dtype({'names':model.paramNames, 'formats':len(r)*[r.dtype.str]})
    
    return r.view(dt)


def goalfcn(indepvars = ''):
    indepvars = [s.strip() for s in indepvars.split(',')]
    
    def wrapfcn(fcn):
        varNames = inspect.getargspec(fcn).args
        paramNames = [v for v in varNames if not v in indepvars]
        #nargs = len(paramNames)
        
        def replfcn(p, *args, **kwargs):
            ar = tuple(p) + args
            return fcn(*ar, **kwargs)
        
        replfcn.paramNames = paramNames
        
        return replfcn
    return wrapfcn
            


def getPhotonNums(colourFilter, metadata):
    nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')

    return nPh

#def eimod(p, n):
#    A, tau = p
#    return A*tau*(exp(-(n-1)/tau) - exp(-n/tau))
    
@goalfcn('n')
def eimod(A, tau, n):
    #A, tau = p
    return A*tau*(exp(-(n-1)/tau) - exp(-n/tau))

@goalfcn('n, dT, nAvg')
def ei2mod(A, tau, n, dT, nAvg=30):
    #A, tau = p
    #print A, tau
    ret =  A*tau*(exp(-sqrt((n-dT)/tau)) - exp(-sqrt(n/tau)))*maximum(1 - n/(dT*nAvg), 0)
    
    #print n, sqrt((n-dT)/tau)

    return ret

@goalfcn('n, dT, nAvg, nDet, tDet')
def ei3mod(A, tau, n, dT, nAvg=30, nDet = 0.7, tDet = .1):#,nDet=.7, tDet = .2):
    #A, tau = p
    #print A, tau
    ret =  A*tau*(exp(-sqrt((n-dT)/tau)) - exp(-sqrt(n/tau)))*(1 + erf((sqrt(maximum(1 - n/(dT*nAvg), 0)) - nDet))/tDet)

    #print n, sqrt((n-dT)/tau)

    return ret

#Decay of event numbers over time
@goalfcn('t, Nm')
def e2mod(A, tau, b, Dt, sT, t, Nm):
    #A, tau, b, Dt, sT = p
    b = 0.5*(1+erf(b))*Nm
    Dt = sqrt(Dt**2 + 1) - 1
    N =  A*exp(-sqrt(t/tau)) + b
    return N*0.5*(1 + erf((Nm + Dt - N)/sT))
#e2mod. = 'A, tau, b, Dt, sT'
    
@goalfcn('t, Nm')
def emod(A, tau, b, Dt, sT, t, Nm):
    #A, tau, b, Dt, sT = p
    b = 0.5*(1+erf(b))*Nm
    Dt = sqrt(Dt**2 + 1) - 1
    N =  A*exp(-(t/tau)) + b**2
    return N*0.5*(1 + erf((Nm + Dt - N)/sT))
    
@goalfcn('t, Nm')
def hmod(A, tau, b, Dt, sT, t, Nm):
    #A, tau, b, Dt, sT = p
    b = 0.5*(1+erf(b))*Nm
    Dt = sqrt(Dt**2 + 1) - 1
    N =  A/(t/tau + 1) + b**2
    return N*0.5*(1 + erf((Nm + Dt - N)/sT))

@goalfcn('t')
def e3mod(A, tau, b, t):
    #A, tau, b = p
    N =  A*exp(-sqrt(t/tau)) + b**2
    return N
    
@goalfcn('t, A')
def e4mod(tau, b, t, A):
    #tau, b = p
    N =  A*exp(-sqrt(t/tau)) + b**2
    return N

#Event amplitudes
@goalfcn('N')
def fImod(A, Ndet, tauDet, tauI, N):
    #A, Ndet, tauDet, tauI = p
    return A*(1 + erf((sqrt(N)-Ndet)/tauDet**2))*exp(-N/tauI)

@goalfcn('N, t, Nco')    
def fITmod(A, Ndet, lamb, tauI, a, Acrit, bg, k, k3, N, t, Nco):
    #A, Ndet, lamb, tauI, a, Acrit, bg, k, k3 = p
    
    #constraints
    Ndet = sqrt(Ndet**2 + 1) - 1 #+ve
    bg = sqrt(bg**2 + 1) - 1
    Acrit = sqrt(Acrit**2 + 1) - 1   
    a = (1+erf(a))/2 # [0,1]
    bg = sqrt(bg**2 + 1) - 1
    k = sqrt(k**2 + 1) - 1
    
    #k2 = sqrt(k2**2 + 1) - 1
    
    r = 1./((t/tauI)**a + 1)
    Ar = A*r
    snr = N/sqrt(N + Ar*Acrit + bg)
    return (N> Nco)*Ar*(1 - exp(-(snr)/Ndet))*(1 + erf((k3 - sqrt(Ar)/k)))*exp(-N/lamb)
    
@goalfcn('N, t, Nco')    
def fITmod2(A, Ndet, lamb, tauI, a, Acrit, bg, N, t, Nco):
    #A, Ndet, lamb, tauI, a, Acrit, bg, k, k3 = p
    
    #constraints
    Ndet = sqrt(Ndet**2 + 1) - 1 #+ve
    bg = sqrt(bg**2 + 1) - 1
    Acrit = sqrt(Acrit**2 + 1) - 1   
    a = (1+erf(a))/2 # [0,1]
    bg = sqrt(bg**2 + 1) - 1
    #k = sqrt(k**2 + 1) - 1
    
    #k2 = sqrt(k2**2 + 1) - 1
    
    r = 1./((t/tauI)**a + 1)
    Ar = A*r
    snr = N/sqrt(N + Ar*Acrit + bg)
    return (N> Nco)*Ar*(1 - exp(-(snr)/Ndet))*exp(-N/lamb)


#########################
#########################
#define decorator to apply each fit fuction independantly over each colour channel
def applyByChannel(fcn):
    args = inspect.getargspec(fcn).args
    def colfcnwrap(pipeline):
        colourFilter = pipeline.colourFilter
        metadata = pipeline.mdh
        chans = colourFilter.getColourChans()
        
        if USE_GUI:
            figure(os.path.split(pipeline.filename)[-1] + ' - ' + fcn.__name__)
    
        if len(chans) == 0:
            fcn(colourFilter, metadata)
        else:
            curChan = colourFilter.currentColour
            
            if 'rng' in args:
                nPh = getPhotonNums(colourFilter, metadata)
                rng = 6*nPh.mean()
    
            chanNames = chans[:]
    
            if 'Sample.Labelling' in metadata.getEntryNames():
                lab = metadata.getEntry('Sample.Labelling')
    
                for i in range(len(lab)):
                    if lab[i][0] in chanNames:
                        chanNames[chanNames.index(lab[i][0])] = lab[i][1]
    
            for ch, i in zip(chans, range(len(chans))):
                colourFilter.setColour(ch)
                PL.ExtendContext({'chan': chanNames[i]})
                if 'rng' in args:
                    fcn(colourFilter, metadata, chanNames[i], i, rng)
                else:
                    fcn(colourFilter, metadata, chanNames[i], i)
                PL.PopContext()
            colourFilter.setColour(curChan)
            
    return colfcnwrap
    
###############
    
def chi2_mse(model, data, r, *args):
    r = r[0]
    f = model(r, *args)
    residual = f-data
    #print (residual.size - r[0].size)
    return (residual**2/f).sum()/(f.size - r.size), (residual**2).mean()
    


@applyByChannel
def fitDecay(colourFilter, metadata, channame='', i=0):
    #get frames in which events occured and convert into seconds
    t = colourFilter['t']*metadata.getEntry('Camera.CycleTime')

    n,bins = histogram(t, 100)

    b1 = bins[:-1]
    Nm = n.max()

    res = FitModel(e2mod, [Nm*2, 15, -3, Nm*3, n[1]/1], n[1:], b1[1:], Nm)
    #mse = (res[2]['fvec']**2).mean()
    #ch2 = chi2(res, n[1:])
    #print ch2
    ch2, mse = chi2_mse(e2mod, n[1:], res, b1[1:], Nm)
    
    PL.AddRecord('/Photophysics/Decay/e2mod', munge_res(e2mod, res, mse=mse, ch2=ch2))
    
    res2 = FitModelPoisson(emod, [Nm*2, 15, -3, Nm*3, n[1]/2], n[1:], b1[1:], Nm)#[0]
    ch2, mse = chi2_mse(hmod, n[1:], res2, b1[1:], Nm)
    
    PL.AddRecord('/Photophysics/Decay/emod', munge_res(emod, res2, mse=mse, ch2=ch2))
    
    res3 = FitModelPoisson(hmod, [Nm*2, 15, -3, Nm*3, n[1]/2], n[1:], b1[1:], Nm)#[0]
    ch2, mse = chi2_mse(hmod, n[1:], res3, b1[1:], Nm)
    
    PL.AddRecord('/Photophysics/Decay/hmod', munge_res(hmod, res3, mse=mse, ch2=ch2))
    
    r4 = FitModelPoisson(e2mod, [Nm*2, 15, -3, Nm*3, n[1]/2], n[1:], b1[1:], Nm)
    ch2, mse = chi2_mse(e2mod, n[1:], r4, b1[1:], Nm)
    PL.AddRecord('/Photophysics/Decay/e2mod_p', munge_res(e2mod, r4, mse=mse, ch2=ch2))

    if USE_GUI:        
        bar(b1/60, n, width=(b1[1]-b1[0])/60, alpha=0.4, fc=colours[i])
        plot(b1/60, e2mod(res[0], b1, Nm), colours[i], lw=3)
        plot(b1/60, emod(res2[0], b1, Nm), colours[i], lw=2, ls='--')
        plot(b1/60, hmod(res3[0], b1, Nm), colours[i], lw=2, ls=':')
        plot(b1/60, e2mod(r4[0], b1, Nm), colours[i], lw=1)
        ylabel('Events')
        xlabel('Acquisition Time [mins]')
        title('Event Rate')
        
        b = 0.5*(1+erf(res[0][2]))*Nm
    
        figtext(.4,.8 -.05*i, channame + '\t$\\tau = %3.2fs,\\;b = %3.2f$' % (res[0][1], b/res[0][0]), size=18, color=colours[i])


@applyByChannel
def fitOnTimes(colourFilter, metadata, channame='', i=0):    
    numPerClump = colourFilter['clumpSize']

    n, bins = histogram(numPerClump, arange(20)+.001)
    n = n/arange(1, 20)

    cycTime = metadata.getEntry('Camera.CycleTime')

    
    res = FitModelWeighted(eimod, [n[0], .2], n[1:], 1./(sqrt(n[1:]) + 1), bins[2:]*cycTime)
    #res = FitModelPoisson(eimod, [n[0], .2], n[1:], bins[2:]*cycTime)
    ch2, mse = chi2_mse(eimod, n[1:], res, bins[2:]*cycTime)
    #mse = (res[2]['fvec']**2).mean()
    res2 = FitModelWeighted(ei2mod, [n[0], .2], n[1:], 1./(sqrt(n[1:]) + 1), bins[2:]*cycTime, cycTime)
    ch22, mse2 = chi2_mse(ei2mod, n[1:], res2, bins[2:]*cycTime, cycTime)
    
    
    PL.AddRecord('/Photophysics/OnTimes/eimod', munge_res(eimod,res, mse=mse))
    PL.AddRecord('/Photophysics/OnTimes/ei2mod', munge_res(ei2mod,res2, mse=mse2))

    #print res[0]

    #figure()

    if USE_GUI:
        semilogy()

        bar(bins[:-1]*cycTime, n, width=cycTime, alpha=0.4, fc=colours[i])

        plot(linspace(1, 20, 50)*cycTime, ei2mod(res2[0], linspace(1, 20, 50)*cycTime, cycTime), colours[i], lw=3, ls='--')
        plot(linspace(1, 20, 50)*cycTime, eimod(res[0], linspace(1, 20, 50)*cycTime), colours[i], lw=3)
        ylabel('Events')
        xlabel('Event Duration [s]')
        ylim((1, ylim()[1]))
        title('Event Duration - CAUTION: unreliable if $\\tau <\\sim$ exposure time')
    
        figtext(.6,.8 -.05*i, channame + '\t$\\tau = %3.4fs$' % (res[0][1], ), size=18, color=colours[i])



@applyByChannel
def fitFluorBrightness(colourFilter, metadata, channame='', i=0, rng = None):
    #nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    #nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
    nPh = getPhotonNums(colourFilter, metadata)
    
    if rng == None:
        rng = nPh.mean()*6
        
    n, bins = histogram(nPh, linspace(0, rng, 100))

    bins = bins[:-1]

    res = FitModel(fImod, [n.max(), bins[n.argmax()]/2, 100, nPh.mean()], n, bins)
    mse = (res[2]['fvec']**2).mean()
    PL.AddRecord('/Photophysics/FluorBrightness/fImod', munge_res(fImod,res, mse=mse))

    #figure()
    #semilogy()
    if USE_GUI:
        bar(bins, n, width=bins[1]-bins[0], alpha=0.4, fc=colours[i])
    
        plot(bins, fImod(res[0], bins), colours[i], lw=3)
        ylabel('Events')
        xlabel('Intensity [photons]')
        #ylim((1, ylim()[1]))
        title('Event Intensity - CAUTION - unreliable if evt. duration $>\\sim$ exposure time')
        #print res[0][2]
    
        figtext(.4,.8 -.05*i, channame + '\t$N_{det} = %3.0f\\;\\lambda = %3.0f$' % (res[0][1], res[0][3]), size=18, color=colours[i])

@applyByChannel    
def fitFluorBrightnessT(colourFilter, metadata, channame='', i=0, rng = None):
    #nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    #nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
    #from mpl_toolkits.mplot3d import Axes3D
    
    nPh = getPhotonNums(colourFilter, metadata)
    t = (colourFilter['t'] - metadata['Protocol.DataStartsAt'])*metadata.getEntry('Camera.CycleTime')
    
    if rng == None:
        rng = nPh.mean()*3
        
    Nco = nPh.min()
        
    n, xbins, ybins = histogram2d(nPh, t, [linspace(0, rng, 50), linspace(0, t.max(), 20)])
    bins = xbins[:-1]
    
    xb = xbins[:-1][:,None]*ones([1,ybins.size - 1])
    yb = ybins[:-1][None, :]*ones([xbins.size - 1, 1])    
    
    res0 = FitModel(fITmod2, [n.max()*3, 1, np.median(nPh), 20, 1e2, 1e2, 100], n, xb, yb, Nco)
    print((res0[0]))
    
    PL.AddRecord('/Photophysics/FluorBrightness/fITmod2', munge_res(fITmod2,res0))
    
    A, Ndet, lamb, tauI, a, Acrit, bg = res0[0]
    #Ndet = Ndet**2
    #NDetM = NDetM**2
    #Acrit = Acrit**2
    #a = (1+erf(a))/2
    
    Ndet = sqrt(Ndet**2 + 1) - 1 #+ve
    bg = sqrt(bg**2 + 1) - 1
    Acrit = sqrt(Acrit**2 + 1) - 1   
    a = (1+erf(a))/2 # [0,1]
    #bg = sqrt(bg**2 + 1) - 1
    #k = sqrt(k**2 + 1) - 1
    
    NDetM = bg
    
    rr = fITmod2(res0[0], xb, yb, Nco)
    if USE_GUI:
    
        figure()
        subplot(131)
        imshow(n, interpolation='nearest')
        colorbar()
        
        subplot(132)
        imshow(rr, interpolation='nearest')
        colorbar()
        
        subplot(133)
        imshow(n - rr, interpolation='nearest')
        colorbar()
        
        title(channame)
        
        figure()
        
        t_ = linspace(t[0], t[-1], 100)
        
        #sc = (lamb/(ybins[1] - ybins[0]))
        #sc = len(ybins)
        sc = 1./(1 - exp(-(ybins[1] - ybins[0])/lamb))
        print(('sc = ', sc))
        y1 = sc*A/((t_/tauI)**a + 1)
        plot(t_, y1)
        plot(t_, sc*(Ndet/((t_/tauI)**a + 1) + NDetM))
        
        bar(ybins[:-1], n.sum(0), width=ybins[1]-ybins[0], alpha=0.5)
        plot(ybins[:-1], rr.sum(0), lw=2)
        
        title(channame)
        xlabel('Time [s]')
        
        figtext(.2,.7 , '$A = %3.0f\\;N_{det} = %3.2f\\;\\lambda = %3.0f\\;\\tau = %3.0f$\n$\\alpha = %3.3f\\;A_{crit} = %3.2f\\;N_{det_0} = %3.2f$' % (A, Ndet, lamb, tauI, a, Acrit, NDetM), size=18)
        
    #    f = figure()
    #    ax = Axes3D(f)
    #    for j in range(len(ybins) - 1):
    #        #print bins[n[:,j].argmax()]
    #        res = FitModel(fImod, [n.max(), bins[n[:,j].argmax()]/2, 100, nPh.mean()], n[:, j], bins)
    #        print res[0]
    #
    #        #figure()
    #        #semilogy()
    #        
    #
    #        ax.bar(bins, n[:,j], zs = j, zdir='y', color = cm.hsv(j/20.), width=bins[1]-bins[0], alpha=0.5)
    #
    #        ax.plot(bins, ones(bins.shape)*j, fImod(res[0], bins), color = array(cm.hsv(j/20.))*.5, lw=3)
    #        
    #    ylabel('Events')
    #    xlabel('Intensity [photons]')
    #    #ylim((1, ylim()[1]))
    #    title(channame)
        #title('Event Intensity - CAUTION - unreliable if evt. duration $>\\sim$ exposure time')
        #print res[0][2]
    
        #figtext(.4,.8 -.05*i, channame + '\t$N_{det} = %3.0f\\;\\lambda = %3.0f$' % (res[0][1], res[0][3]), size=18, color=colours[i])
        
        #t = colourFilter['t']*metadata.getEntry('Camera.CycleTime')
    


