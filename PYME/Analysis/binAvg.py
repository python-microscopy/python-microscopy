#!/usr/bin/python

##################
# correlationCoeffs.py
#
# Copyright David Baddeley, 2010
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

def binned_average(binVar, indepVar, edges):
    """
    Take the binned average of values in indepVar binned according to their corresponding entries in binVar and a given set of edges.

    Parameters
    ----------
    binVar : ndarray 
        array which is used, in combination with `edges`, to determine bin membership.
    indepVar : ndarray
        array of the independent variable, to be averaged within each bin. Must be the same size as binVar.
    edges : ndarray
        array of bin edges

    Returns
    -------
    bn : int ndarray
        Number of elements within each bin
    bm : float ndarray
        Mean value of indepVar within each bin
    bs : float ndarray
        Standard deviation of indepVar within each bin

    """
    num_bins = len(edges) - 1
    
    # initialize outputs with zero'd arrays
    bm = np.zeros(num_bins)
    bs = np.zeros(num_bins)
    bn = np.zeros(num_bins, dtype='i')

    # loop over each bin
    for i, el, er in zip(range(num_bins), edges[:-1], edges[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        # outputs are zero-initialized:  we only need to modify the output if there are elements of binVar in this bin
        if bn[i] != 0:
            bm[i] = v.mean()
            bs[i] = v.std()

    return bn, bm, bs
    
#for backwards compatibility
binAvg = binned_average
    
def binned_median(binVar, indepVar, edges):
    """
        Take the binned median of values in indepVar binned according to their corresponding entries in binVar and a given set of edges.
        
        Parameters
        ----------
        binVar : ndarray
            array used, in combination with `edges`, to determine bin membership.
        indepVar : ndarray 
            array of the independent variable, to have the median calculated within each bin. Must be the same size as binVar.
        edges : ndarray
            array of bin edges

        Returns
        -------
        bn : int ndarray
            Number of pixels within each bin
        bm : float ndarray
            Median value of indepVar within each bin
        bs : float ndarray
            Standard deviation of indepVar within each bin

    """
    num_bins = len(edges) - 1
    # initialize outputs with zero'd arrays
    bm = np.zeros(num_bins)
    bs = np.zeros(num_bins)
    bn = np.zeros(num_bins, dtype='i')

    for i, el, er in zip(range(num_bins), edges[:-1], edges[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        # outputs are zero-initialized:  we only need to modify the output if there are elements of binVar in this bin
        if bn[i] != 0:
            bm[i] = np.median(v)
            bs[i] = v.std()

    return bn, bm, bs

#backwards compatibility
binMedian=binned_median

def errorPlot(filter, bins):
    import matplotlib.pyplot as plt
    x = (bins[:-1] + bins[1:])/2.

    a1 = plt.axes()

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_x0'], bins)

    a1.plot(x, bm, lw=2, c='b', label='x')
    a1.fill_between(x, np.maximum(bm-bs, 0), bm + bs, facecolor='b', alpha=0.2)

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_y0'], bins)

    a1.plot(x, bm, lw=2, c='g', label='y')
    a1.fill_between(x, np.maximum(bm-bs, 0), bm + bs, facecolor='g', alpha=0.2)

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_z0'], bins)

    a1.plot(x, bm, lw=2, c='r', label='z')
    a1.fill_between(x, np.maximum(bm-bs, 0), bm + bs, facecolor='r', alpha=0.2)

    plt.ylabel('Fit Error [nm]')
    plt.xlabel('Defocus [nm]')
    plt.legend()
    plt.ylim(0, 120)


    a2 = plt.twinx()

    a2.plot(x, bn, 'k')
    plt.ylabel('Number of fitted events')