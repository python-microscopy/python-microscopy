#!/usr/bin/python
##################
# splitterFilters.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import pylab
import numpy as np
import scipy.misc

import re

from PYME.misc.spectraComputations import spectrum
from PYME.misc import scrapeSemrock
from PYME.misc import scrapeOmega
from PYME.misc import scrapeChroma

omegaFNames = scrapeOmega.getFilterNames()
semrockFNames = scrapeSemrock.getFilterNames()
chromaFNames = scrapeChroma.getFilterNames()

omegaDichroics = [fn for fn in omegaFNames if 'DR' in fn and re.search(r'[67]\d\d', fn)]
omegaDichroics.sort()
semrockDichroics = [fn for fn in semrockFNames if 'Di' in fn and re.search(r'[67]\d\d', fn)]
semrockDichroics.sort()

semrockDichroics = ['FF741-Di01', 'FF700-Di01']
omegaDichroics = ['708DRLP']

#
#chromaDichroics = [f for f in chromaFNames if ('dc' in f or 'DC' in f) and re.search(r'[67]\d\d', f)]
#chromaDichroics.sort()
chromaDichroics = []

omegaBlockers = [fn for fn in omegaFNames if (not 'DR' in fn) and re.search(r'[67]\d\d', fn)]
omegaBlockers.sort()

omegaBlockers = []

semrockBlockers = [fn for fn in semrockFNames if (not 'Di' in fn) and re.search(r'[67]\d\d', fn)]
semrockBlockers.sort()
semrockBlockers = []
#print len(semrockBlockers)

#chromaBlockers = [f for f in chromaFNames if not ('dc' in f or 'DC' in f or 'bs' in f) and re.search(r'\w(6[5-9]|7[0-6])\d', f)]
#chromaBlockers.sort()
chromaBlockers = ['HQ710/50m', 'HQ775/50x']

#semrockBlockers = semrockBlockers[:10]


#dyes = ['Alexa Fluor 647', 'Alexa Fluor 680', 'Alexa Fluor 700']#, 'Alexa Fluor 750']#, 'Alexa Fluor 750']
dyes = ['Alexa Fluor 680', 'Alexa Fluor 700', 'Alexa Fluor 750']

blockingFilter = (spectrum(scrapeOmega.getFilterSpectrum('690ALP'))/100)#*spectrum(scrapeChroma.getFilterSpectrum('Q680LP'))
#blockingFilter = spectrum(scrapeSemrock.getFilterSpectrum('LP02-647RU'))

dyeSpectra = [(spectrum(scrapeOmega.getDyeSpectrum(dyeName)[0])*blockingFilter).norm() for dyeName in dyes]

import scipy.stats
dyeSpectra[1].magnitude[:] = scipy.stats.norm.pdf(dyeSpectra[1].wavelengths, 719, 18)
dyeSpectra[0].magnitude[:] = scipy.stats.norm.pdf(dyeSpectra[0].wavelengths, 702, 18)

ts = np.zeros((len(dyes), len(omegaDichroics) + len(semrockDichroics) + len(chromaDichroics) + 1, len(omegaBlockers) + len(semrockBlockers) + len(chromaBlockers) + 1), 'f4')
rs = np.zeros_like(ts)

i = 0

#straight 50:50 beamsplitter (or PRI)
f1 = .5
rf1 = 1 - f1

for j in range(len(dyes)):
    ts[j,i, 0] = (dyeSpectra[j]*f1).sum()
    rs[j,i, 0] = (dyeSpectra[j]*rf1).sum()

    k = 1
    for bfn in omegaBlockers:
        f2 = spectrum(scrapeOmega.getFilterSpectrum(bfn))/100
        ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
        rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
        k +=1

    for bfn in semrockBlockers:
        f2 = spectrum(scrapeSemrock.getFilterSpectrum(bfn))
        try: #work around for FF01-446/523/600/677 which has a bung spectrum filename
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
        except:
            pass

        k +=1

    for bfn in chromaBlockers:
        f2 = spectrum(scrapeChroma.getFilterSpectrum(bfn))
        ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
        rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()

        k +=1

i+=1

for fn in omegaDichroics:
    f1 = spectrum(scrapeOmega.getFilterSpectrum(fn))/100
    rf1 = 1 - f1

    for j in range(len(dyes)):
        ts[j,i, 0] = (dyeSpectra[j]*f1).sum()
        rs[j,i, 0] = (dyeSpectra[j]*rf1).sum()

        k = 1
        for bfn in omegaBlockers:
            f2 = spectrum(scrapeOmega.getFilterSpectrum(bfn))/100
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
            k +=1

        for bfn in semrockBlockers:
            f2 = spectrum(scrapeSemrock.getFilterSpectrum(bfn))
            try: #work around for FF01-446/523/600/677 which has a bung spectrum filename
                ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
                rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
            except:
                pass
            
            k +=1

        for bfn in chromaBlockers:
            f2 = spectrum(scrapeChroma.getFilterSpectrum(bfn))
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()

            k +=1

    i+=1

for fn in semrockDichroics:
    f1 = spectrum(scrapeSemrock.getFilterSpectrum(fn))
    rf1 = 1 - f1

    for j in range(len(dyes)):
        ts[j,i, 0] = (dyeSpectra[j]*f1).sum()
        rs[j,i, 0] = (dyeSpectra[j]*rf1).sum()

        k = 1
        for bfn in omegaBlockers:
            f2 = spectrum(scrapeOmega.getFilterSpectrum(bfn))/100
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
            k +=1

        for bfn in semrockBlockers:
            f2 = spectrum(scrapeSemrock.getFilterSpectrum(bfn))
            try:
                ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
                rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
            except:
                pass

            k +=1

        for bfn in chromaBlockers:
            f2 = spectrum(scrapeChroma.getFilterSpectrum(bfn))
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()

            k +=1
        

    i+=1

for fn in chromaDichroics:
    f1 = spectrum(scrapeChroma.getFilterSpectrum(fn))
    rf1 = 1 - f1

    for j in range(len(dyes)):
        ts[j,i, 0] = (dyeSpectra[j]*f1).sum()
        rs[j,i, 0] = (dyeSpectra[j]*rf1).sum()

        k = 1
        for bfn in omegaBlockers:
            f2 = spectrum(scrapeOmega.getFilterSpectrum(bfn))/100
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
            k +=1

        for bfn in semrockBlockers:
            f2 = spectrum(scrapeSemrock.getFilterSpectrum(bfn))
            try:
                ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
                rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()
            except:
                pass

            k +=1

        for bfn in chromaBlockers:
            f2 = spectrum(scrapeChroma.getFilterSpectrum(bfn))
            ts[j,i, k] = (dyeSpectra[j]*f1*f2).sum()
            rs[j,i, k] = (dyeSpectra[j]*rf1*f2).sum()

            k +=1


    i+=1


#pylab.figure()
#a = pylab.axes([.1,.2,.85,.7])
#for j in range(len(dyes)):
#    pylab.plot(rs[j,:, :], marker = 'x', label=dyes[j])
#
#pylab.xticks(range(rs.shape[1]))
#a.set_xticklabels(omegaDichroics + semrockDichroics, rotation='vertical')
#legend()
#pylab.draw()

ratios = np.zeros((rs.shape[0], rs.shape[1], rs.shape[2], rs.shape[2]), 'f4')
throughput = np.zeros_like(ratios)

for k in range(rs.shape[2]):
    for l in range(rs.shape[2]):
        throughput[:,:,k, l] = rs[:,:,k]+ts[:,:,l]
        ratios[:,:,k, l] = rs[:,:,k]/throughput[:,:,k, l]
#rs/(rs + ts)
#throughput = rs + ts


minThroughput = throughput.min(0)
minRatioDist = np.ones(ratios.shape[1:], 'f4')

#ratioDists = np.zeros((scipy.misc.comb(rs.shape[0], 2, True), rs.shape[1], rs.shape), 'f4')

k = 0
for i in range(rs.shape[0]):
    for j in range(i+1, rs.shape[0]):
        #ratioDists[k, :] = np.abs(ratios[i, :] - ratios[j, :])
        k+= 1
        #if not i == j:
        ndist = np.abs(ratios[i, :, :, :] - ratios[j, :, :, :])
        ndist[np.isnan(ndist)] = 0
        minRatioDist = np.minimum(minRatioDist, ndist)

#pylab.figure()
#a = pylab.axes([.1,.2,.85,.7])
#pylab.plot(ratioDists.min(0), c='k', lw=2)
#pylab.plot(ratioDists.T, marker='x')
#pylab.xticks(range(rs.shape[1]))
#a.set_xticklabels(omegaDichroics + semrockDichroics, rotation='vertical')
#pylab.draw()


c = minThroughput*minRatioDist
msk = (minRatioDist > .1)*(minThroughput > .3)
d_i, t_i, r_i = np.mgrid[:c.shape[0], :c.shape[1], :c.shape[2]]

dichroicNames = ['50:50'] + omegaDichroics + semrockDichroics + chromaDichroics
filterNames = ['none'] + omegaBlockers + semrockBlockers + chromaBlockers

#def onpick(event):
#    n = event.ind
#    d, t, r = d_i[msk][n], t_i[msk][n], r_i[msk][n]
#    print dichroicNames[d], filterNames[t], filterNames[r]
#
#    print ratios[:, d,t,r], throughput[:,d,t,r]
#
#f = pylab.figure()
#
#pylab.scatter(minThroughput[msk], minRatioDist[msk], c = d_i[msk], s = 5, linewidth=0, picker=5)
#f.canvas.mpl_connect('pick_event', onpick)

from PYME.misc import filterPlots
filterPlots.plotFilterScatter(throughput, ratios, minThroughput[msk], minRatioDist[msk], d_i[msk], t_i[msk], r_i[msk], dichroicNames, filterNames, ['708DRLP', 'FF700-Di01', 'FF741-Di01', '810DCSP', 'HQ785/60', 'HQ775/55'])

