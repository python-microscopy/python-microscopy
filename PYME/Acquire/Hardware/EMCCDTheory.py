#!/usr/bin/python

##################
# EMCCDTheory.py
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

from numpy import *

def FSquared(M, N):
    """Excess noise factor as a function of multiplication (M) and number of
       Stages (N)

       From Robbins and Hadwen, 2002, IEEE Trans. Electon Dev."""

    return 2*(M-1)*M**(-(float(N)+1)/float(N)) + 1/M

def SNR(S, ReadNoise, M, N, B = 0):
    return (S-B)/sqrt((ReadNoise/M)**2 + FSquared(M, N)*S)

def M(V, Vbr, T, N, n=2):
    """em gain as a function of voltage (V), breakdown voltage, temperature (T), and
       number of gain stages (N). 2 < n < 6 is an emperical exponent."""

    return (1./(1. - (V/(Vbr*(((T + 273.)/300.)**0.2)))**n))**N