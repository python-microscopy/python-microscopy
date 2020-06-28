#!/usr/bin/python
##################
# expectationMaximisation.py
#
# Copyright David Baddeley, 2011
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
import numpy as np
#import numexpr as ne

def em(data, initialMeans, initialSigs, nIters=10):
    nClasses = len(initialMeans)

    means = np.array(initialMeans, 'f')
    vars = np.array(initialSigs, 'f')**2
    pis = np.ones(nClasses)/nClasses

    weights = np.ones((len(data), nClasses))/nClasses

    #print means, vars, pis, weights.mean(0)

    for iter in range(nIters):
        #Expectation Step
        weights = (pis/np.sqrt(2*np.pi*vars))[None, :]*np.exp(-(data[:,None]-means[None,:])**2/((2*vars)[None,:]))
        weights = weights/(weights.sum(1)[:,None])
        
        #Maximisation step
        means = (weights*data[:,None]).sum(0)/weights.sum(0)
        vars = (weights*(data[:,None]-means[None,:])**2).sum(0)/weights.sum(0)
        pis = weights.sum(0)/len(data)

        #print iter, means, vars, pis, weights.mean(0)


    return means, np.sqrt(vars), pis, weights

def emw(data, errors, initialMeans, initialSigs, nIters=10, updateSigs=True):
    nClasses = len(initialMeans)

    means = np.array(initialMeans, 'f')[None, :]
    vars = (np.array(initialSigs, 'f')**2)[None, :]
    pis = (np.ones(nClasses)/nClasses)[None, :]

    #mVars = vars

    weights = np.ones((len(data), nClasses))/nClasses
    data = data[:,None]
    dataVars = (errors**2)[:,None]

    #print means, vars, pis, weights.mean(0)

    for iter in range(nIters):
        #Expectation Step
        weights = (pis/np.sqrt(2*(vars + dataVars)))*np.exp(-(data -means)**2/(2*(vars + dataVars)))
        #weights = ne.evaluate('(pis/sqrt(2*(vars + dataVars)))*exp(-(data -means)**2/(2*(vars + dataVars)))')
        weights = weights/(weights.sum(1)[:,None])

        #print weights/dataVars[:,None]

        #if the probability of one species falls too low, discard it
        pis = (weights.sum(0)/len(data))
        M = pis > .4/pis.size
        #print M, vars[:,M]

        weights = weights[:,M]/(weights[:,M].sum(1)[:,None])
        vars = vars[:,M]

        #Maximisation step
        means = ((weights*(data/dataVars)).sum(0)/(weights/dataVars).sum(0))[None, :]
        if updateSigs:
            nm = 2*(dataVars + vars)
            vars = np.maximum((weights*((data-means)**2 - dataVars)/nm).sum(0)/(weights/nm).sum(0), 1)[None, :]
        pis = (weights.sum(0)/len(data))[None, :]

        #print means

        I = np.argsort(pis.squeeze())
        #print I
        
        M = np.ones(I.size)

        for i in range(I.size - 1):
            ind = I[i]
            #print means.shape, I[i]
            if (np.abs(means[0,ind] - means[0,I[(i+1):]]) < np.sqrt(vars[0,ind])).any():
                #print i, M.shape
                M[ind] = 0

        M = M > 0.5

        #print M

        means = means[:,M]
        vars = vars[:,M]
        pis = pis[:,M]

        #mVars = ((weights*(data/dataVars)).sum(0)/(weights/dataVars).sum(0))[None, :]

        #print means



        #print iter, means, vars, pis, weights.mean(0), weights.sum(0)

    mVars = 1./((weights*(1/dataVars)).sum(0))




    return np.atleast_1d(means.squeeze()), np.atleast_1d(np.sqrt(vars.squeeze())), np.atleast_1d(pis.squeeze()), np.atleast_1d(np.sqrt(mVars.squeeze())), weights

def emwv(data, errors, initialMeans, initialSigs, nIters=10, updateSigs=True):
    nClasses = initialMeans.shape[1]
    nVars = data.shape[1]

    means = np.array(initialMeans, 'd')[None, :,:]
    vars = (np.array(initialSigs, 'd')**2)[None, :,:]
    pis = (np.ones((nClasses))/nClasses)[None,None,:]

    #mVars = vars

    weights = np.ones((data.shape[0], nClasses))/nClasses
    data = data.astype('d')[:,:,None]
    dataVars = (errors.astype('d')**2)[:,:,None]

    #print means, vars, pis, weights.mean(0)

    #print means.shape,vars.shape, pis.shape, data.shape, dataVars.shape

    #print pis*means

    #print np.isnan(data).sum(), np.isnan(dataVars).sum()

    for iter in range(nIters):
        #Expectation Step
        weights = ((pis/np.sqrt(2*(vars + dataVars)))*np.exp(-(data -means)**2/(2*(vars + dataVars)))).prod(1)
        #weights = ne.evaluate('(pis/sqrt(2*(vars + dataVars)))*exp(-(data -means)**2/(2*(vars + dataVars)))')
        weights = (weights/(weights.sum(1)[:,None]))

        #print weights/dataVars[:,None]

        #print weights.shape, np.isnan(weights).sum(), data.shape

        #if the probability of one species falls too low, discard it
        pis = (weights.sum(0)/data.shape[0]).squeeze()
        #print pis
        M = pis > 0 #.4/pis.size
        #print M
        #print M, vars

        weights = (weights[:,M]/(weights[:,M].sum(1)[:,None]))[:,None,:]
        #print weights.shape
        vars = vars[:,:,M]

        #print vars

        #Maximisation step
        means = ((weights*(data/dataVars)).sum(0)/(weights/dataVars).sum(0))[None,:, :]
        #print means
        if updateSigs:
            nm = 2*(dataVars + vars)
            vars = np.maximum((weights*((data-means)**2 - dataVars)/nm).sum(0)/(weights/nm).sum(0), 1)[None, :, :]
        pis = ((weights.sum(0)/len(data)).squeeze())[None,None, :]

        #print means, vars, pis

#        I = np.argsort(pis.squeeze())
        #print I

#        M = np.ones(I.size)
#
#        for i in range(I.size - 1):
#            ind = I[i]
#            #print means.shape, I[i]
#            if (np.abs(means[0,ind] - means[0,I[(i+1):]]) < 1.5*np.sqrt(vars[0,ind])).any():
#                #print i, M.shape
#                M[ind] = 0
#
#        M = M > 0.5
#
#        #print M
#
#        means = means[:,M]
#        vars = vars[:,M]
#        pis = pis[:,M]

        #mVars = ((weights*(data/dataVars)).sum(0)/(weights/dataVars).sum(0))[None, :]

        #print means



        #print iter, means, vars, pis, weights.mean(0), weights.sum(0)

    mVars = 1./((weights*(1/dataVars)).sum(0))




    return np.atleast_1d(means.squeeze()), np.atleast_1d(np.sqrt(vars.squeeze())), np.atleast_1d(pis.squeeze()), np.atleast_1d(np.sqrt(mVars.squeeze())), weights

def emwg(data, sigs, nIters=10, width=5, nPerClass=10, updateSigs=False):
    return emw(data, sigs, np.linspace(data.min(), data.max(), data.size/nPerClass), width*np.ones(data.size/nPerClass), nIters, updateSigs)


def plotRes(data, errors, r):
    # import pylab
    import matplotlib.pyplot as plt
    import matplotlib.cm
    plt.figure()

    nObs = len(data)

    n, bins, patches = plt.hist(data, 2*np.sqrt(nObs), fc=[.7,.7,.7])

    binSize = bins[1] - bins[0]
    x = np.arange(bins[0], bins[-1])
    

    means, sigs, pis, mVars, weights = r

    inds = np.argmax(weights, 1)

    for i in range(means.size):
        #print i
        c = matplotlib.cm.hsv(float(i)/means.size)
        n, bin_s, patches = plt.hist(data[inds == i], bins, alpha=0.3, facecolor=c)
    
    ys = np.zeros_like(x)

    i = 0
    for m, s, p in zip(means, sigs, pis):
        c = matplotlib.cm.hsv(float(i)/means.size)
        y = nObs*p*binSize*np.exp(-(x-m)**2/(2*s**2))/np.sqrt(2*np.pi*s**2)
        ys += y

        i+= 1

        plt.plot(x,y, lw=2, color=c)

    #plt.plot(x, ys, lw=3)

    plt.figure()

    ci = (r[4]*np.arange(r[0].size)[None,:]).sum(1)

    I = np.argsort(ci)
    cis = ci[I]
    cil = 0

    for i in range(means.size):
        c = matplotlib.cm.hsv(float(i)/means.size)

        print(c)

        plt.axvline(means[i], color=c)
        plt.axvspan(means[i] - sigs[i], means[i] + sigs[i], alpha=0.5, facecolor=c)

        cin = cis.searchsorted(i+0.5)

        plt.axhspan(cil, cin, alpha=0.3, facecolor=c)

        cil = cin

    plt.errorbar(data[I], np.arange(data.size), xerr=errors[I], fmt='.')



    










    



        
