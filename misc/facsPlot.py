#!/usr/bin/python
##################
# facsPlot.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from pylab import *
from scipy import ndimage

def facsPlotScatter(x, y, nbins=None, ms=1):
    if nbins == None:
        nbins = 0.25*sqrt(len(x))
    n, xedge, yedge = histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

    dx = diff(xedge[:2])
    dy = diff(yedge[:2])

    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])

    scatter(x, y, c=c, s=ms, edgecolors='none')

def facsPlotScatterLog(x, y, nbins=None, ms=1):
    if nbins == None:
        nbins = 0.25*sqrt(len(x))
    n, xedge, yedge = histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

    dx = diff(xedge[:2])
    dy = diff(yedge[:2])

    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])

    scatter(x, y, c=c, s=ms, edgecolors='none')

    ax = gca()
    ax.get_xaxis().set_scale('log')
    ax.get_yaxis().set_scale('log')
    draw()
    
    

def facsPlotContour(x, y, nbins=None):
    if nbins == None:
        nbins = 0.25*sqrt(len(x))
    n, xedge, yedge = histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

#    dx = diff(xedge[:2])
#    dy = diff(yedge[:2])
#
#    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])
#
#    scatter(x, y, c=c, s=ms, edgecolors='none')
    #X, Y = meshgrid(xedge, yedge)

    contour((xedge[:-1] + xedge[1:])/2, (yedge[:-1] + yedge[1:])/2, log(n.T + .1), 5, cmap = cm.copper_r)
