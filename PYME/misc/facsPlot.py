#!/usr/bin/python
##################
# facsPlot.py
#
# Copyright David Baddeley, 2009
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

#from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def facsPlotScatter(x, y, nbins=None, ms=1):
    if nbins is None:
        nbins = 0.25*np.sqrt(len(x))
    n, xedge, yedge = np.histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

    dx = np.diff(xedge[:2])
    dy = np.diff(yedge[:2])

    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])

    plt.scatter(x, y, c=c, s=ms, edgecolors='none')
    
def facsPlotScatter2(x, y, nxbins=None, nybins=None, ms=1):
    #if nbins == None:
    #    nbins = 0.25*sqrt(len(x))
    n, xedge, yedge = np.histogram2d(x, y, bins = [nxbins,nybins], range=[(min(x), max(x)), (min(y), max(y))])

    dx = np.diff(xedge[:2])
    dy = np.diff(yedge[:2])

    #c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])

    bx = np.floor((x - xedge[0])/dx).astype('i')
    by = np.floor((y - yedge[0])/dy).astype('i')
    
    c = n[bx, by]

    plt.scatter(x, y, c=c, s=ms, edgecolors='none')

def facsPlotScatterLog(x, y, nbins=None, ms=1):
    if nbins is None:
        nbins = 0.25*np.sqrt(len(x))
    n, xedge, yedge = np.histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

    dx = np.diff(xedge[:2])
    dy = np.diff(yedge[:2])

    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])

    plt.scatter(x, y, c=c, s=ms, edgecolors='none')

    ax = plt.gca()
    ax.get_xaxis().set_scale('log')
    ax.get_yaxis().set_scale('log')
    plt.draw()
    
    

def facsPlotContour(x, y, nbins=None):
    if nbins is None:
        nbins = 0.25*np.sqrt(len(x))
    n, xedge, yedge = np.histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

#    dx = diff(xedge[:2])
#    dy = diff(yedge[:2])
#
#    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])
#
#    scatter(x, y, c=c, s=ms, edgecolors='none')
    #X, Y = meshgrid(xedge, yedge)

    plt.contour((xedge[:-1] + xedge[1:])/2, (yedge[:-1] + yedge[1:])/2, np.log(n.T + .1), 5, cmap = plt.cm.copper_r)
