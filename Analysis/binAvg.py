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
from pylab import *

def binAvg(binVar, indepVar, bins):
    bm = np.zeros(len(bins) - 1)
    bs = np.zeros(len(bins) - 1)
    bn = np.zeros(len(bins) - 1, dtype='i')

    for i, el, er in zip(range(len(bm)), bins[:-1], bins[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        if bn[i] == 0:
            bm[i] = 0
            bs[i] = 0
        else:
            bm[i] = v.mean()
            bs[i] = v.std()

    return bn, bm, bs
    
def binMedian(binVar, indepVar, bins):
    bm = np.zeros(len(bins) - 1)
    bs = np.zeros(len(bins) - 1)
    bn = np.zeros(len(bins) - 1, dtype='i')

    for i, el, er in zip(range(len(bm)), bins[:-1], bins[1:]):
        v = indepVar[(binVar >= el)*(binVar < er)]

        bn[i] = len(v)
        if bn[i] == 0:
            bm[i] = 0
            bs[i] = 0
        else:
            bm[i] = np.median(v)
            bs[i] = v.std()

    return bn, bm, bs


def errorPlot(filter, bins):
    x = (bins[:-1] + bins[1:])/2.

    a1 = axes()

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_x0'], bins)

    a1.plot(x, bm, lw=2, c='b', label='x')
    a1.fill_between(x, maximum(bm-bs, 0), bm + bs, facecolor='b', alpha=0.2)

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_y0'], bins)

    a1.plot(x, bm, lw=2, c='g', label='y')
    a1.fill_between(x, maximum(bm-bs, 0), bm + bs, facecolor='g', alpha=0.2)

    bn, bm, bs = binAvg(filter['fitResults_z0'], filter['fitError_z0'], bins)

    a1.plot(x, bm, lw=2, c='r', label='z')
    a1.fill_between(x, maximum(bm-bs, 0), bm + bs, facecolor='r', alpha=0.2)

    ylabel('Fit Error [nm]')
    xlabel('Defocus [nm]')
    legend()
    ylim(0, 120)


    a2 = twinx()

    a2.plot(x, bn, 'k')
    ylabel('Number of fitted events')