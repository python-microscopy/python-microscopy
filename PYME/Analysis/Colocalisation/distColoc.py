#!/usr/bin/python

##################
# distColoc.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.Analysis.points import DistHist
from pylab import *
#from PYME.Acquire.ExecTools import execBG


#def calcCollHist(colourFilter, colour):

def rebin(a, fact):
    b = a[::fact]
    for i in range(1, fact):
        b += a[i::fact]
    return b

def calcDistCorr(cfilter, colourA, colourB, nbins=100, binsize=10):
    '''Do pointwise / ripley based colocalisation between colourA and colourB'''
    #save current state of colour filter
    old_col = cfilter.currentColour

    #grab the x and y vals in each channel
    cfilter.setColour(colourA)
    xA = cfilter['x']
    yA = cfilter['y']

    cfilter.setColour(colourB)
    xB = cfilter['x']
    yB = cfilter['y']

    #generate randomly distributed x & y values with the same density as colourB
    xR = (xB.max() - xB.min())*rand(len(xB)) + xB.min()
    yR = (yB.max() - yB.min())*rand(len(xB)) + yB.min()

    hAA = DistHist.distanceHistogram(xA, yA, xA, yA, nbins, binsize)
    hAB = DistHist.distanceHistogram(xA, yA, xB, yB, nbins, binsize)
    hBB = DistHist.distanceHistogram(xB, yB, xB, yB, nbins, binsize)
    hBA = DistHist.distanceHistogram(xB, yB, xA, yA, nbins, binsize)

    hAR = DistHist.distanceHistogram(xA, yA, xR, yR, nbins, binsize)
    hBR = DistHist.distanceHistogram(xB, yB, xR, yR, nbins, binsize)
    
    d = binsize*arange(nbins)

    figure()
    #subplot(211)
    plot(d, hAA/hAA.sum()-hAR/hAR.sum(), label='%s' % (colourA,), lw=2)
    plot(d, hAB/hAB.sum()-hBR/hBR.sum(), label='%s' % (colourB,), lw=2)
    plot(d, hAA/hAA.sum()-hAB/hAB.sum(), '--',label='%s - %s' % (colourA, colourB), lw=2)

    grid()
    legend()
    xlabel('Radius [nm] from %s' % colourA)
    ylabel('Excess Labelling')

    figure()
    #subplot(211)
    plot(d, hBB/hBB.sum()-hBR/hBR.sum(), label='%s' % (colourB,), lw=2)
    plot(d, hBA/hBA.sum()-hAR/hAR.sum(), label='%s' % (colourA,), lw=2)
    plot(d, hBB/hBB.sum()-hBA/hBA.sum(), '--',label='%s - %s' % (colourB, colourA), lw=2)

    grid()
    legend()
    xlabel('Radius [nm] from %s' % colourB)
    ylabel('Excess Labelling')

    #subplot(212)
    figure()

    corr = 1 - abs(hAB/hAB.sum()-hAA/hAA.sum()).sum()/abs(hAA/hAA.sum()-hAR/hAR.sum()).sum()

    print 'Correlation Factor: %3.2f' % corr

    ccorr = 1 - cumsum(abs(hAB/hAB.sum()-hAA/hAA.sum()))/cumsum(abs(hAA/hAA.sum()-hAR/hAR.sum()))
    plot(d,ccorr)
    ylim(-1,1)
    grid()
    ylabel('Correlation Factor')
    xlabel('Radius [nm] from %s' % colourA)

    figure()

    corr = 1 - abs(hBA/hBA.sum()-hBB/hBB.sum()).sum()/abs(hBB/hBB.sum()-hBR/hBR.sum()).sum()

    print 'Correlation Factor: %3.2f' % corr

    ccorr = 1 - cumsum(abs(hBA/hBA.sum()-hBB/hBB.sum()))/cumsum(abs(hBB/hBB.sum()-hBR/hBR.sum()))
    plot(d,ccorr)
    ylim(-1,1)
    grid()
    ylabel('Correlation Factor')
    xlabel('Radius [nm] from %s' % colourB)

    #title('%s vs %s  - Correlation Factor = %3.2f' % (colourB, colourA, corr))



    #restore state of colour filter
    cfilter.setColour(old_col)

