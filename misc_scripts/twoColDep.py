#!/usr/bin/python

##################
# twoColDep.py
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
#import scipy as sp
from PYME.localization import ofind
from PYME.localization.FitFactories.LatGaussFitFRTC import FitFactory, FitResultsDType
from PYME.Analysis import MetaData

def fitDep(g,r,ofindThresh, dx, dy):
    rg = r + g #detect objects in sum image
    
    ofd = ofind.ObjectIdentifier(rg)

    ofd.FindObjects(ofindThresh, blurRadius=2)

    res_d = np.empty(len(ofd), FitResultsDType)

    class foo:
        pass

    md = MetaData.TIRFDefault

    md.chroma = foo()
    md.chroma.dx = dx
    md.chroma.dy = dy

    ff = FitFactory(np.concatenate((g.reshape(512, -1, 1), r.reshape(512, -1, 1)),2), md)

    
    for i in range(len(ofd)):    
        p = ofd[i]
        res_d[i] = ff.FromPoint(round(p.x), round(p.y))
        

    return res_d
