#!/usr/bin/python
##################
# expectationMaximisation.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np

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

def emw(data, errors, initialMeans, initialSigs, nIters=10):
    nClasses = len(initialMeans)

    means = np.array(initialMeans, 'f')
    vars = np.array(initialSigs, 'f')**2
    pis = np.ones(nClasses)/nClasses

    weights = np.ones((len(data), nClasses))/nClasses
    dataVars = errors**2

    #print means, vars, pis, weights.mean(0)

    for iter in range(nIters):
        #Expectation Step
        weights = (pis[None, :]/np.sqrt(2*(vars[None,:] + dataVars[:,None])))*np.exp(-(data[:,None]-means[None,:])**2/(2*(vars[None,:] + dataVars[:,None])))
        weights = weights/(weights.sum(1)[:,None])

        #print weights/dataVars[:,None]

        #Maximisation step
        means = (weights*(data/dataVars)[:,None]).sum(0)/(weights/dataVars[:,None]).sum(0)
        nm = 2*(dataVars[:,None] + vars[None,:])
        vars = np.maximum(1*(weights*((data[:,None]-means[None,:])**2 - dataVars[:,None])/nm).sum(0)/(weights/nm).sum(0), 1)
        pis = weights.sum(0)/len(data)

        #print iter, means, vars, pis, weights.mean(0), weights.sum(0)


    return means, np.sqrt(vars), pis, weights


def plotRes(data, r):
    import pylab
    pylab.figure()

    nObs = len(data)

    n, bins, patches = pylab.hist(data, 2*np.sqrt(nObs))

    binSize = bins[1] - bins[0]
    x = np.arange(bins[0], bins[-1])
    

    means, sigs, pis, weights = r

    inds = np.argmax(weights, 1)

    for i in range(len(means)):
        #print i
        n, bin_s, patches = pylab.hist(data[inds == i], bins, alpha=0.3)
    
    ys = np.zeros_like(x)

    for m, s, p in zip(means, sigs, pis):
        y = nObs*p*binSize*np.exp(-(x-m)**2/(2*s**2))/np.sqrt(2*np.pi*s**2)
        ys += y

        pylab.plot(x,y, lw=2)

    pylab.plot(x, ys, lw=3)









    



        
