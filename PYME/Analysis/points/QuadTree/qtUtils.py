#!/usr/bin/python

##################
# qtUtils.py
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

import pointQT
from PYME.Analysis import MetaData

def compareOriginalWithRec(ofd, qt, radius = 250, md = MetaData.TIRFDefault):
    I_origs = []
    for p in ofd:
        I_origs.append(ofd.filteredData[p.x,p.y])

    N_points = []
    for p in ofd:
        Ns.append(len(pointQT.getInRadius(qt, p.x*md.voxelsize_nm.x, p.y*md.voxelsize_nm.y, radius)))

    return (I_origs, N_points)
