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
import numpy as np


#def calcCollHist(colourFilter, colour):

def rebin(a, fact):
    b = a[::fact]
    for i in range(1, fact):
        b += a[i::fact]
    return b

def calcDistCorr(cfilter, colourA, colourB, nbins=100, binsize=10):
    '''Do pointwise / ripley based colocalisation between colourA and colourB'''
    
    import matplotlib.pyplot as plt
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
    xR = (xB.max() - xB.min())*np.random.rand(len(xB)) + xB.min()
    yR = (yB.max() - yB.min())*np.random.rand(len(xB)) + yB.min()

    hAA = DistHist.distanceHistogram(xA, yA, xA, yA, nbins, binsize)
    hAB = DistHist.distanceHistogram(xA, yA, xB, yB, nbins, binsize)
    hBB = DistHist.distanceHistogram(xB, yB, xB, yB, nbins, binsize)
    hBA = DistHist.distanceHistogram(xB, yB, xA, yA, nbins, binsize)

    hAR = DistHist.distanceHistogram(xA, yA, xR, yR, nbins, binsize)
    hBR = DistHist.distanceHistogram(xB, yB, xR, yR, nbins, binsize)
    
    d = binsize*np.arange(nbins)

    plt.figure()
    #subplot(211)
    plt.plot(d, hAA/hAA.sum()-hAR/hAR.sum(), label='%s' % (colourA,), lw=2)
    plt.plot(d, hAB/hAB.sum()-hBR/hBR.sum(), label='%s' % (colourB,), lw=2)
    plt.plot(d, hAA/hAA.sum()-hAB/hAB.sum(), '--',label='%s - %s' % (colourA, colourB), lw=2)

    plt.grid()
    plt.legend()
    plt.xlabel('Radius [nm] from %s' % colourA)
    plt.ylabel('Excess Labelling')

    plt.figure()
    #subplot(211)
    plt.plot(d, hBB/hBB.sum()-hBR/hBR.sum(), label='%s' % (colourB,), lw=2)
    plt.plot(d, hBA/hBA.sum()-hAR/hAR.sum(), label='%s' % (colourA,), lw=2)
    plt.plot(d, hBB/hBB.sum()-hBA/hBA.sum(), '--',label='%s - %s' % (colourB, colourA), lw=2)

    plt.grid()
    plt.legend()
    plt.xlabel('Radius [nm] from %s' % colourB)
    plt.ylabel('Excess Labelling')

    #subplot(212)
    plt.figure()

    corr = 1 - np.abs(hAB/hAB.sum()-hAA/hAA.sum()).sum()/np.abs(hAA/hAA.sum()-hAR/hAR.sum()).sum()

    print('Correlation Factor: %3.2f' % corr)

    ccorr = 1 - np.cumsum(np.abs(hAB/hAB.sum()-hAA/hAA.sum()))/np.cumsum(np.abs(hAA/hAA.sum()-hAR/hAR.sum()))
    plt.plot(d,ccorr)
    plt.ylim(-1,1)
    plt.grid()
    plt.ylabel('Correlation Factor')
    plt.xlabel('Radius [nm] from %s' % colourA)

    plt.figure()

    corr = 1 - np.abs(hBA/hBA.sum()-hBB/hBB.sum()).sum()/np.abs(hBB/hBB.sum()-hBR/hBR.sum()).sum()

    print('Correlation Factor: %3.2f' % corr)

    ccorr = 1 - np.cumsum(np.abs(hBA/hBA.sum()-hBB/hBB.sum()))/np.cumsum(np.abs(hBB/hBB.sum()-hBR/hBR.sum()))
    plt.plot(d,ccorr)
    plt.ylim(-1,1)
    plt.grid()
    plt.ylabel('Correlation Factor')
    plt.xlabel('Radius [nm] from %s' % colourB)

    #title('%s vs %s  - Correlation Factor = %3.2f' % (colourB, colourA, corr))



    #restore state of colour filter
    cfilter.setColour(old_col)

