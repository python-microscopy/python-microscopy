#!/usr/bin/python

###############
# coverslipGeometries.py
#
# Copyright David Baddeley, 2012
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
################
from scipy.optimize import fmin
import numpy as np




class geom:
    def __init__(self, fixedParameters):
        self.__dict__.update(fixedParameters)
        #self.fitableParameters = fitableParameters

    def Fit(self, x,y):
        self.fittedParameters = fmin(self.errFunc, self.genStartParams(x, y), args=(x, y))


class circGeom(geom):
    def errFunc(self, params, x, y):
        x0, y0, margin = params

        return (np.abs((x - x0)**2 + (y - y0)**2 - (self.r - margin)**2)).sum()

    def genStartParams(self, x,y):
        x0 = x.mean()
        y0 = y.mean()

        return [x0, y0, 0]


GEOMTYPES = {
'10mm round' : (circGeom, {'r':5})
}

def GetGeometry(type):
    cls, args = GEOMTYPES[type]
    return cls(args)



