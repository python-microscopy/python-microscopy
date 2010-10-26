#!/usr/bin/python

##################
# locify.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
# Generates a series of point/fluorophore positions from a given grayscale image
#
##################

from numpy.random import rand
import numpy as np


def locify(im, pixelSize=1, pointsPerPixel=0.1):
    '''Create a set of point positions with a density corresponding to the
    input image im. Useful for generating localisation microscopy images from
    conventional images. Assumes im is a 2D array with values between 0 and 1
    and interprets this value as a probability. pointsPerPixel gives the point density for a prob. of 1.'''

    #what range shold we generate points in
    xmax = im.shape[0]
    ymax = im.shape[1]

    #generate a number of candidate points based on uniform labelling
    #which will be accepted/rejected later
    numPoints = im.shape[0]*im.shape[1]*pointsPerPixel

    x = xmax*rand(numPoints) - .5
    y = ymax*rand(numPoints) - .5

    #index into array to get probability of acceptance
    p = im[x.round().astype('i'), y.round().astype('i')]

    #use monte-carlo to accept points with the given probability
    mcInd = rand(len(x)) < p

    #take subset of positions and scale to pixel size
    x = pixelSize*x[mcInd]
    y = pixelSize*y[mcInd]

    return (x,y)
    

def testPattern():
    '''generate a test pattern'''
    pass
    
