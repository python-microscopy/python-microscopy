from pylab import *
from PYME.Analysis._fithelpers import *
from PYME.Analysis.DeClump import deClump
from scipy.special import erf

colours = ['r', 'g', 'b']

def eimod(p, n):
    A, tau = p
    return A*tau*(exp(-(n-1)/tau) - exp(-n/tau)) 

def e2mod(p, t):
    A, tau, b = p
    return A*exp(-(t**.5)/tau) + b

def fImod(p, N):
    A, Ndet, tauDet, tauI = p
    return A*(1 + erf((sqrt(N)-Ndet)/tauDet))*exp(-N/tauI)


def fitDecayChan(colourFilter, metadata, channame='', i=0):
    #get frames in which events occured and convert into seconds
    t = colourFilter['t']*metadata.getEntry('Camera.CycleTime')

    n,bins = histogram(t, 100)

    b1 = bins[:-1]

    res = FitModel(e2mod, [n[0], 5, 10], n, b1)
    bar(b1/60, n, width=(b1[1]-b1[0])/60, alpha=0.4, fc=colours[i])
    plot(b1/60, e2mod(res[0], b1), colours[i], lw=3)
    ylabel('Events')
    xlabel('Acquisition Time [mins]')
    title('Event Rate')

    figtext(.4,.8 -.05*i, channame + '\t$\\tau = %3.2fs,\\;b = %3.2f$' % (res[0][1], res[0][2]/res[0][0]), size=18, color=colours[i])

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
    clumpIndices = deClump.findClumps(colourFilter['t'].astype('i'), colourFilter['x'].astype('f4'), colourFilter['y'].astype('f4'), colourFilter['error_x'].astype('f4'), 3)
    numPerClump, b = histogram(clumpIndices, arange(clumpIndices.max() + 1.5) + .5)

    n, bins = histogram(numPerClump, arange(20)+.001)

    n = n[:10]
    bins = bins[:10]

    cycTime = metadata.getEntry('Camera.CycleTime')

    res = FitModel(eimod, [n[0], .05], n[:9], bins[1:]*cycTime)

    #figure()

    semilogy()
    
    bar(bins*cycTime, n, width=cycTime, alpha=0.4, fc=colours[i])

    plot(linspace(1, 10, 50)*cycTime, eimod(res[0], linspace(1, 10, 50)*cycTime), colours[i], lw=3)
    ylabel('Events')
    xlabel('Event Duration [s]')
    ylim((1, ylim()[1]))
    title('Event Duration - CAUTION: unreliable if $\\tau <\\sim$ exposure time')

    figtext(.6,.8 -.05*i, channame + '\t$\\tau = %3.4fs$' % (res[0][1], ), size=18, color=colours[i])


def fitOnTimes(colourFilter, metadata):
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
            fitOnTimesChan(colourFilter, metadata, chanNames[i], i)
        colourFilter.setColour(curChan)


def fitFluorBrightnessChan(colourFilter, metadata, channame='', i=0):
    nPh = (colourFilter['A']*2*math.pi*(colourFilter['sig']/(1e3*metadata.getEntry('voxelsize.x')))**2)
    nPh = nPh*metadata.getEntry('Camera.ElectronsPerCount')/metadata.getEntry('Camera.TrueEMGain')
    
    n, bins = histogram(nPh, linspace(0, nPh.mean()*6, 100))

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

    figtext(.4,.8 -.05*i, channame + '\t$N_{det} = %3.0f\\;\\lambda = %3.0f$' % (res[0][1], res[0][3]), size=18, color=colours[i])


def fitFluorBrightness(colourFilter, metadata):
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
            fitFluorBrightnessChan(colourFilter, metadata, chanNames[i], i)
        colourFilter.setColour(curChan)
