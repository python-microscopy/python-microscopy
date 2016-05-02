#!/usr/bin/python

##################
# spark.py
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

from scipy import *

def sparkMod(p, X,T):
    maxP, Td, Tr, x0, t0, sig0, b = p

    f = ((T - t0) > 0)*maxP*exp(-(T-t0)/Td)*(1 - exp(-(T-t0)/Tr))*exp(-((X-x0)**2)/(2*(sig0*(1 + (((T - t0) > 0)*(T - t0))**.25)**2)))+ b

    return f
