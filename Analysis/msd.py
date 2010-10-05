import numpy as np
#from PYME.Analysis.binAvg import binAvg

#Boltzman's constant
Kb = 1.38e-23

def msd(x, y, t, tbins=1e3):
    '''calculate the msd of a trace. Use PYME.Analysis.DistHist.msdHistogram instead'''
    dists = np.zeros(tbins)
    ns = np.zeros(tbins, dtype='i')
    tbins = np.linspace(0, (t.max() - t.min() + 1), tbins +1)

    tdelta = tbins[1]

    for i in range(len(x)):
        tdist = np.abs(t - t[i])

        rdists = (x - x[i])**2 + (y-y[i])**2

        #bn, bm, bs = binAvg(tdist, rdists, tbins)

        #dists += bm
        for t_, r_ in zip(tdist, rdists):
            j = np.floor(t_/tdelta)
            #print j
            dists[j] += r_
            ns[j] += 1

    return tbins, dists/ns

def stokesEinstein(radius, viscoscity=.001, dimensions=3, temperature=293):
    '''calculate the expected diffusion coeficient for a spherical particle of a
    given radius. All parameters in SI units.'''
    return Kb*temperature/(2*dimensions*viscoscity*radius)
