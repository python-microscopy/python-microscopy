#!/usr/bin/python
##################
# annealThresh.py
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
from scipy import ndimage

def genDefaultMask(nDim):
    """default mask is a n-dimensional donut around point"""
    m = np.ones([3]*nDim, 'f')
    m.__setitem__((1,)*nDim,0)

    return m
    
def gen2DMask(nDim):
    """default mask is a n-dimensional donut around point"""
    m = np.ones([3]*nDim, 'f')
    m.__setitem__((1,)*nDim,0)
    
    m[:,:,0] = 0
    m[:,:,2] = 0

    return m

def annealThresh(image, valP50, valPslope=1, neighP50=4, neighSlope=.25, mask=None, nIters=10):
    """segment an image based on both intensity and neighbourbood relationships"""

    #if no mask then use default
    if mask is None:
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
    """segment an image based on both intensity and neighbourbood relationships"""

    #if no mask then use default
    if mask is None:
        neighbourMask = genDefaultMask(image.ndim)
    else:
        neighbourMask = mask

    #print neighbourMask.shape

    nNeighbours = neighbourMask.sum()

    neighP50 *= nNeighbours
    neighSlope /= nNeighbours

    #calculate probability that a pixel belongs to object based on it's intensity
    imP = (image/(2*valP50) - 0.5)*valPslope + 0.5

    if out is None:
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
    
def annealThresh2D(image, valP50, valPslope=1, neighP50=.5, neighSlope=1, mask=None, nIters=10, out = None):
    """segment an image based on both intensity and neighbourbood relationships"""

    #if no mask then use default
    if mask is None:
        neighbourMask = gen2DMask(image.ndim)
    else:
        neighbourMask = mask

    #print neighbourMask.shape

    nNeighbours = neighbourMask.sum()

    neighP50 *= nNeighbours
    neighSlope += nNeighbours

    #calculate probability that a pixel belongs to object based on it's intensity
    imP = (image/(2*valP50) - 0.5)*valPslope + 0.5

    if out is None:
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