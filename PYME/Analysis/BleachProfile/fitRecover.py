#!/usr/bin/python

###############
# fitRecover.py
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
"""
Fit the recovery of a fluorophore from the dark state. Use with the PYMEAcquire protocol "recover671.py" which switches the
laser on and off in pulses to allow bleaching steps followed by recovery steps of varying length. Frame numbers in this file
match those in the protocol.

Probably sufferring from bitrot.
"""
from . import intensProf
from . import kinModels
from PYME.Analysis._fithelpers import FitModel
import numpy as np


def FitTrace(tr, mdh):
    # from pylab import *
    import matplotlib.pyplot as plt
    #tr = (ds - mdh.getEntry('Camera.ADOffset')).sum(1).sum(0)

    cycTime = mdh.getEntry('Camera.CycleTime')
    t = cycTime*np.arange(len(tr))

    plt.figure()
    plt.plot(t, tr)

    yp = -.1*tr.max()

    y1 = tr[61:80]

    r1 = FitModel(kinModels.e3mod, [y1[0], 1.0, 0], y1, cycTime*np.arange(len(y1)))

    plt.plot(t[61:80], kinModels.e3mod(r1[0], cycTime*np.arange(len(y1))), lw=2)
    plt.text(t[65], yp, '%3.2fs'%r1[0][1])

    y2 = tr[81:1500]
    r2 = FitModel(intensProf.eMod5, [y2[-1], -50, 50], y2, cycTime*np.arange(len(y2)))

    plt.plot(t[81:1500], intensProf.eMod5(r2[0], cycTime*np.arange(len(y2))), lw=2)
    plt.text(t[700], yp, '%3.2fs'%r2[0][2])

    y3 = tr[1501:1600]

    r3 = FitModel(kinModels.e3mod, [y3[0], 1.0, 0], y3, cycTime*np.arange(len(y3)))

    plt.plot(t[1501:1600], kinModels.e3mod(r3[0], cycTime*np.arange(len(y3))), lw=2)
    plt.text(t[1520], yp, '%3.2fs'%r3[0][1])

#    y4 = tr[1601:2500]
#    r4 = FitModel(intensProf.eMod5, [y4[-1], -50, 50], y4, cycTime*arange(len(y4)))
#
#    plot(t[1601:2500], intensProf.eMod5(r4[0], cycTime*arange(len(y4))), lw=2)
#    text(t[1900], yp, '$\\tau = %3.2fs$'%r4[0][2])

    y5 = tr[2502:2600]
    r5 = FitModel(kinModels.e3mod, [y5[0], 1.0, 0], y5, cycTime*np.arange(len(y5)))

    plt.plot(t[2502:2600], kinModels.e3mod(r5[0], cycTime*np.arange(len(y5))), lw=2)
    plt.text(t[2520], yp, '%3.2fs'%r5[0][1])
    
    y6 = tr[3502:3600]
    r6 = FitModel(kinModels.e3mod, [y6[0], 1.0, 0], y6, cycTime*np.arange(len(y6)))

    plt.plot(t[3502:3600], kinModels.e3mod(r6[0], cycTime*np.arange(len(y6))), lw=2)
    plt.text(t[3520], yp, '%3.2fs'%r6[0][1])





