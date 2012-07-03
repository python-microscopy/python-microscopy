#!/usr/bin/python

##################
# piecewise.py
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


def piecewiseLinear(t, tvals, grads):
    tvals = numpy.hstack([0., tvals, 1e9])

    res = numpy.zeros(t.shape)

    a = numpy.hstack((0, numpy.cumsum(numpy.diff(tvals)*grads)))

    for i in range(len(grads)):
        res += (t >= tvals[i])*(t <  tvals[i+1])*(a[i] + grads[i]*(t-tvals[i]))

    return res