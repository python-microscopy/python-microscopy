#!/usr/bin/python

##################
# msd.py
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

import numpy as np
#from PYME.Analysis.binAvg import binAvg

#Boltzman's constant
Kb = 1.38e-23

def msd(x, y, t, tbins=1e3):
    """calculate the msd of a trace. Use PYME.Analysis.points.DistHist.msdHistogram instead"""
    dists = np.zeros(tbins)
    ns = np.zeros(tbins, dtype='i')
    tbins = np.linspace(0, (t.max() - t.min() + 1), tbins +1)

    tdelta = tbins[1]

    for i in range(len(x)):
        tdist = np.abs(t - t[i])

        rdists = (x - x[i])**2 + (y-y[i])**2

        #bn, bm, bs = binAvg(tdist, rdists, tbins)

        #dists += bm
        for t_, r_ in zip(tdist, rdists):
            j = np.floor(t_/tdelta)
            #print j
            dists[j] += r_
            ns[j] += 1

    return tbins, dists/ns

def stokesEinstein(radius, viscoscity=.001, dimensions=3, temperature=293):
    """calculate the expected diffusion coeficient for a spherical particle of a
    given radius. All parameters in SI units."""
    return Kb*temperature/(2*dimensions*viscoscity*radius)
