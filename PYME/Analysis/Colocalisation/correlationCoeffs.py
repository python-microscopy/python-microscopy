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

def pearson(X, Y):
    X = X - X.mean()
    Y = Y-Y.mean()
    return (X*Y).sum()/np.sqrt((X*X).sum()*(Y*Y).sum())

def overlap(X, Y):
    return (X*Y).sum()/np.sqrt((X*X).sum()*(Y*Y).sum())


def thresholdedManders(A, B, tA, tB):
    '''Manders, as practically used with threshold determined masks'''

    MA = ((B > tB)*A).sum()/A.sum()
    MB = ((A > tA)*B).sum()/B.sum()

    return MA, MB