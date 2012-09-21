#!/usr/bin/python

##################
# thresholding.py
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

import numpy

def isodata_f(img, nbins=255):
    '''calculate isodata threshold - does iterations on histogrammed data rather than
    raw data to improve speed
    
    img - the image data
    nbins - the number of bins used for the histogram
    '''

    N, bins = numpy.histogram(img, nbins)
    
#    #calculate bin centres
#    bin_mids = 0.5*(bins[:-1] + bins[1:])
    
    #precalculate bin weightings
    bw = N*numpy.arange(len(N))

    #start off with the largest possible delta
    delta = nbins

    #start off with threshold at middle of range
    t = delta/2

    while delta > 0:
        #new threshold = mean of the two segment means
        tn = (bw[:t].sum()/N[:t].sum() + bw[t:].sum()/N[t:].sum())/2

        delta = abs(tn - t)
        t = tn

    return bins[t]


def isodata(img, tol = 1e-3):
    '''correct isodata implementation - slower than above
    tolerance indicates fraction of data range when delta is deemed not to have changed
    '''

    imin = img.min()
    imax = img.max()

    #initial delta is data range
    delta = imax - imin

    tol = tol*delta

    #initial threshold is range midpoint
    t = imin + delta/2.

    while delta > tol:
        tn = (img[img <= t].mean() + img[img > t].mean())/2.
        delta = abs(t - tn)
        tn = t

    return t

def signalFraction(img, frac, nbins = 5000):
    ''' threshold to include a certain fraction of the total signal'''

    N, bins = numpy.histogram(img, nbins)

    #calculate bin centres
    bin_mids = (bins[:-1] )

    cN = numpy.cumsum(N*bin_mids)

    #print cN

    i = numpy.argmin(abs(cN - cN[-1]*(1-frac)))

    #print abs(cN - cN[-1]*frac)
    #print i, frac

    return bins[i]
