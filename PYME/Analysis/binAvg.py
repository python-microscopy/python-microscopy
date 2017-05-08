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
import matplotlib.pyplot as plt

def binAvg(binVar, indepVar, bins):
    """

    Parameters
    ----------
    binVar : array to be binned using 'bins' input. Each element of this array corresponds to an element in indepVar
    indepVar : array of the independent variable, to be averaged within each bin
    bins : array of bin edges

    Returns
    -------
    bn : Number of pixels within each bin
    bm : Mean value of indepVar within each bin
    bs : Standard deviation of indepVar within each bin

    """
    bl = len(bins) - 1
    # initialize outputs with zero'd arrays
    bm = np.zeros(bl)
    bs = np.zeros(bl)
    bn = np.zeros(bl, dtype='i')

    # loop over each bin
    for i, el, er in zip(range(bl), bins[:-1], bins[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        # outputs are zero-initialized:  we only need to modify the output if there are elements of binVar in this bin
        if bn[i] != 0:
            bm[i] = v.mean()
            bs[i] = v.std()

    return bn, bm, bs
    
def binMedian(binVar, indepVar, bins):
    """

        Parameters
        ----------
        binVar : array to be binned using 'bins' input. Each element of this array corresponds to an element in indepVar
        indepVar : array of the independent variable, to be averaged within each bin
        bins : array of bin edges

        Returns
        -------
        bn : Number of pixels within each bin
        bm : Median value of indepVar within each bin
        bs : Standard deviation of indepVar within each bin

    """
    bl = len(bins) - 1
    # initialize outputs with zero'd arrays
    bm = np.zeros(bl)
    bs = np.zeros(bl)
    bn = np.zeros(bl, dtype='i')

    for i, el, er in zip(range(bl), bins[:-1], bins[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        # outputs are zero-initialized:  we only need to modify the output if there are elements of binVar in this bin
        if bn[i] != 0:
            bm[i] = np.median(v)
            bs[i] = v.std()

    return bn, bm, bs


def errorPlot(filter, bins):
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