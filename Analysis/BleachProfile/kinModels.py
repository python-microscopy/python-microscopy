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

colours = ['r', 'g', 'b']

def getPhotonNums(colourFilter, metadata):
    nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')

    return nPh

def eimod(p, n):
    A, tau = p
    return A*tau*(exp(-(n-1)/tau) - exp(-n/tau))

def ei2mod(p, n, dT, nAvg=30):
    A, tau = p
    #print A, tau
    ret =  A*tau*(exp(-sqrt((n-dT)/tau)) - exp(-sqrt(n/tau)))*maximum(1 - n/(dT*nAvg), 0)
    
    #print n, sqrt((n-dT)/tau)

    return ret

def ei3mod(p, n, dT, nAvg=30, nDet = 0.7, tDet = .1):#,nDet=.7, tDet = .2):
    A, tau = p
    #print A, tau
    ret =  A*tau*(exp(-sqrt((n-dT)/tau)) - exp(-sqrt(n/tau)))*(1 + erf((sqrt(maximum(1 - n/(dT*nAvg), 0)) - nDet))/tDet)

    #print n, sqrt((n-dT)/tau)

    return ret

#Decay of event numbers over time
def e2mod(p, t):
    A, tau, b, Dt, sT = p
    N =  A*exp(-sqrt(t/tau)) + b**2
    return N*(1 + erf((Dt - N)/sT))

def e3mod(p, t):
    A, tau, b = p
    N =  A*exp(-sqrt(t/tau)) + b**2
    return N

def e4mod(p, t, A):
    tau, b = p
    N =  A*exp(-sqrt(t/tau)) + b**2
    return N

#Event amplitudes
def fImod(p, N):
    A, Ndet, tauDet, tauI = p
    return A*(1 + erf((sqrt(N)-Ndet)/tauDet**2))*exp(-N/tauI)
    
def fITmod(p, N, t):
    A, Ndet, lamb, tauI, a, Acrit, NDetM = p
    
    #constraints
    Ndet = Ndet**2 #+ve
    NDetM = NDetM**2
    Acrit = Acrit**2   
    a = (1+erf(a))/2 # [0,1]
    
    r = 1./((t/tauI)**a + 1)
    Ar = A*r
    return Ar*(1 + erf((N-(Ndet*r - NDetM)*(1 + Ar/Acrit))/N))*exp(-N/lamb)


def fitDecayChan(colourFilter, metadata, channame='', i=0):
    #get frames in which events occured and convert into seconds
    t = colourFilter['t']*metadata.getEntry('Camera.CycleTime')

    n,bins = histogram(t, 100)

    b1 = bins[:-1]

    res = FitModel(e2mod, [n[1], 5, 10, n[1], n[1]/10], n[1:], b1[1:])
    bar(b1/60, n, width=(b1[1]-b1[0])/60, alpha=0.4, fc=colours[i])
    plot(b1/60, e2mod(res[0], b1), colours[i], lw=3)
    ylabel('Events')
    xlabel('Acquisition Time [mins]')
    title('Event Rate')

    figtext(.4,.8 -.05*i, channame + '\t$\\tau = %3.2fs,\\;b = %3.2f$' % (res[0][1], res[0][2]**2/res[0][0]), size=18, color=colours[i])

def fitDecay(colourFilter, metadata):
    chans = colourFilter.getColourChans()
    
    figure()

    if len(chans) == 0:
        fitDecayChan(colourFilter, metadata)
    else:
        curChan = colourFilter.currentColour

        chanNames = chans[:]

        if 'Sample.Labelling' in metadata.getEntryNames():
            lab = metadata.getEntry('Sample.Labelling')

            for i in range(len(lab)):
                if lab[i][0] in chanNames:
                    chanNames[chanNames.index(lab[i][0])] = lab[i][1]

        for ch, i in zip(chans, range(len(chans))):
            colourFilter.setColour(ch)
            fitDecayChan(colourFilter, metadata, chanNames[i], i)
        colourFilter.setColour(curChan)

def fitOnTimesChan(colourFilter, metadata, channame='', i=0):
    #if 'error_x' in colourFilter.keys():
    #    clumpIndices = deClump.findClumps(colourFilter['t'].astype('i'), colourFilter['x'].astype('f4'), colourFilter['y'].astype('f4'), colourFilter['error_x'].astype('f4'), 3)
    #else:
    #    clumpIndices = deClump.findClumps(colourFilter['t'].astype('i'), colourFilter['x'].astype('f4'), colourFilter['y'].astype('f4'), 30*ones(len(colourFilter['x'])).astype('f4'), 3)

    #numPerClump, b = histogram(clumpIndices, arange(clumpIndices.max() + 1.5) + .5)
    
    numPerClump = colourFilter['clumpSize']

    n, bins = histogram(numPerClump, arange(20)+.001)
    n = n/arange(1, 20)

    #n = n[:10]
    #bins = bins[:10]

    cycTime = metadata.getEntry('Camera.CycleTime')

    semilogy()

    bar(bins[:-1]*cycTime, n, width=cycTime, alpha=0.4, fc=colours[i])

    #print (1./(sqrt(n) + 1))

    #res = FitModelWeighted(ei2mod, [n[0], .2], n[1:], 1./(sqrt(n[1:]) + 1), bins[2:]*cycTime, cycTime)
    res = FitModelWeighted(eimod, [n[0], .2], n[1:], 1./(sqrt(n[1:]) + 1), bins[2:]*cycTime)

    #print res[0]

    #figure()

    

    #plot(linspace(1, 20, 50)*cycTime, ei2mod(res[0], linspace(1, 20, 50)*cycTime, cycTime), colours[i], lw=3)
    plot(linspace(1, 20, 50)*cycTime, eimod(res[0], linspace(1, 20, 50)*cycTime), colours[i], lw=3)
    ylabel('Events')
    xlabel('Event Duration [s]')
    ylim((1, ylim()[1]))
    title('Event Duration - CAUTION: unreliable if $\\tau <\\sim$ exposure time')

    figtext(.6,.8 -.05*i, channame + '\t$\\tau = %3.4fs$' % (res[0][1], ), size=18, color=colours[i])


def fitOnTimes(colourFilter, metadata):
    chans = colourFilter.getColourChans()

    figure()

    if len(chans) == 0:
        fitOnTimesChan(colourFilter, metadata)
    else:
        curChan = colourFilter.currentColour

        chanNames = chans[:]

        if 'Sample.Labelling' in metadata.getEntryNames():
            lab = metadata.getEntry('Sample.Labelling')

            for i in range(len(lab)):
                if lab[i][0] in chanNames:
                    chanNames[chanNames.index(lab[i][0])] = lab[i][1]

        for ch, i in zip(chans, range(len(chans))):
            colourFilter.setColour(ch)
            fitOnTimesChan(colourFilter, metadata, chanNames[i], i)
        colourFilter.setColour(curChan)


def fitFluorBrightnessChan(colourFilter, metadata, channame='', i=0, rng = None):
    #nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    #nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
    nPh = getPhotonNums(colourFilter, metadata)
    
    if rng == None:
        rng = nPh.mean()*6
        
    n, bins = histogram(nPh, linspace(0, rng, 100))

    bins = bins[:-1]

    res = FitModel(fImod, [n.max(), bins[n.argmax()]/2, 100, nPh.mean()], n, bins)

    #figure()
    #semilogy()

    bar(bins, n, width=bins[1]-bins[0], alpha=0.4, fc=colours[i])

    plot(bins, fImod(res[0], bins), colours[i], lw=3)
    ylabel('Events')
    xlabel('Intensity [photons]')
    #ylim((1, ylim()[1]))
    title('Event Intensity - CAUTION - unreliable if evt. duration $>\\sim$ exposure time')
    #print res[0][2]

    figtext(.4,.8 -.05*i, channame + '\t$N_{det} = %3.0f\\;\\lambda = %3.0f$' % (res[0][1], res[0][3]), size=18, color=colours[i])
    
def fitFluorBrightnessTChan(colourFilter, metadata, channame='', i=0, rng = None):
    #nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    #nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
    from mpl_toolkits.mplot3d import Axes3D
    
    nPh = getPhotonNums(colourFilter, metadata)
    t = (colourFilter['t'] - metadata['Protocol.DataStartsAt'])*metadata.getEntry('Camera.CycleTime')
    
    if rng == None:
        rng = nPh.mean()*3
        
    n, xbins, ybins = histogram2d(nPh, t, [linspace(0, rng, 50), linspace(0, t.max(), 20)])
    bins = xbins[:-1]
    
    xb = xbins[:-1][:,None]*ones([1,ybins.size - 1])
    yb = ybins[:-1][None, :]*ones([xbins.size - 1, 1])    
    
    res0 = FitModel(fITmod, [n.max()*3, sqrt(bins[n[:,0].argmax()]/2), nPh.mean(), 200, 1, 7, 1], n, xb, yb)
    print res0[0]
    
    A, Ndet, lamb, tauI, a, Acrit, NDetM = res0[0]
    Ndet = Ndet**2
    NDetM = NDetM**2
    Acrit = Acrit**2
    a = (1+erf(a))/2
    
    rr = fITmod(res0[0], xb, yb)
    
    
    figure()
    subplot(131)
    imshow(n, interpolation='nearest')
    
    subplot(132)
    imshow(rr, interpolation='nearest')
    
    subplot(133)
    imshow(n - rr, interpolation='nearest')
    
    title(channame)
    
    figure()
    
    t_ = linspace(t[0], t[-1], 100)
    
    sc = (lamb/(ybins[1] - ybins[0]))
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
    


def fitFluorBrightness(colourFilter, metadata):
    chans = colourFilter.getColourChans()

    figure()

    if len(chans) == 0:
        fitFluorBrightnessChan(colourFilter, metadata)
    else:
        #nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
        #nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
        nPh = getPhotonNums(colourFilter, metadata)
        
        rng = 6*nPh.mean()

        curChan = colourFilter.currentColour

        chanNames = chans[:]

        if 'Sample.Labelling' in metadata.getEntryNames():
            lab = metadata.getEntry('Sample.Labelling')

            for i in range(len(lab)):
                if lab[i][0] in chanNames:
                    chanNames[chanNames.index(lab[i][0])] = lab[i][1]

        for ch, i in zip(chans, range(len(chans))):
            colourFilter.setColour(ch)
            fitFluorBrightnessChan(colourFilter, metadata, chanNames[i], i, rng)
        colourFilter.setColour(curChan)

def fitFluorBrightnessT(colourFilter, metadata):
    chans = colourFilter.getColourChans()

    #figure()

    if len(chans) == 0:
        fitFluorBrightnessTChan(colourFilter, metadata)
    else:
        #nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
        #nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
        #nPh = getPhotonNums(colourFilter, metadata)
        
        #rng = 6*nPh.mean()

        curChan = colourFilter.currentColour

        chanNames = chans[:]

        if 'Sample.Labelling' in metadata.getEntryNames():
            lab = metadata.getEntry('Sample.Labelling')

            for i in range(len(lab)):
                if lab[i][0] in chanNames:
                    chanNames[chanNames.index(lab[i][0])] = lab[i][1]

        for ch, i in zip(chans, range(len(chans))):
            colourFilter.setColour(ch)
            fitFluorBrightnessTChan(colourFilter, metadata, chanNames[i], i)
        colourFilter.setColour(curChan)