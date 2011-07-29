#!/usr/bin/python
##################
# annealThresh.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import numpy as np
from scipy import ndimage

def genDefaultMask(nDim):
    '''default mask is a n-dimensional donut around point'''
    m = np.ones([3]*nDim, 'f')
    m.__setitem__((1,)*nDim,0)

    return m

def annealThresh(image, valP50, valPslope=1, neighP50=4, neighSlope=.25, mask=None, nIters=10):
    '''segment an image based on both intensity and neighbourbood relationships'''

    #if no mask then use default
    if mask == None:
        neighbourMask = genDefaultMask(image.ndim)
    else:
        neighbourMask = mask

    #print neighbourMask.shape

    #calculate probability that a pixel belongs to object based on it's intensity
    imP = (image/(2*valP50) - 0.5)*valPslope + 0.5

    #create initial object segmentation
    obj = imP > 0.5

    #now do the simulated annealing bit
    for i in range(nIters):
        #calculate probability of pixel given neighbours
        pNeigh = (ndimage.convolve(obj.astype('f'), neighbourMask) - neighP50)*neighSlope

        #total probability is sum of neighbour and intensity based probabilities
        pObj = imP + pNeigh

        #create our new estimate of the object
        obj = np.random.random(imP.shape) < pObj


    return obj, pObj


def annealThresh2(image, valP50, valPslope=1, neighP50=.5, neighSlope=1, mask=None, nIters=10, out = None):
    '''segment an image based on both intensity and neighbourbood relationships'''

    #if no mask then use default
    if mask == None:
        neighbourMask = genDefaultMask(image.ndim)
    else:
        neighbourMask = mask

    #print neighbourMask.shape

    nNeighbours = neighbourMask.sum()

    neighP50 *= nNeighbours
    neighSlope += nNeighbours

    #calculate probability that a pixel belongs to object based on it's intensity
    imP = (image/(2*valP50) - 0.5)*valPslope + 0.5

    if out == None:
        out = np.zeros_like(image)

    #create initial object segmentation

    #print out.shape, imP.shape

    out[:] = (imP > 0.5)[:]

    #now do the simulated annealing bit
    for i in range(nIters):
        #calculate probability of pixel given neighbours
        pNeigh = (ndimage.convolve(out.astype('f'), neighbourMask) - neighP50)*neighSlope

        #total probability is sum of neighbour and intensity based probabilities
        pObj = imP + pNeigh

        #create our new estimate of the object
        out[:] =  pObj > .5


    return out