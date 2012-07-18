#!/usr/bin/python

###############
# pSpaceMap.py
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
xvs = 70*arange(-2.01, 2,.1)
yvs = 70*arange(-2.01, 2,.1)
zvs = arange(-500, 500, 100)

PsfFitCSIR.setModel(seriesName, None)

misf = zeros([len(xvs), len(yvs), len(zvs)])
m0 = PsfFitCSIR.f_Interp3d([1,0,0,-300,0], X, Y, Z)
for i in range(len(xvs)):
    for j in range(len(yvs)):
        for k in range(len(zvs)):
            misf[i,j, k] = ((m0 -PsfFitCSIR.f_Interp3d([1,xvs[i],yvs[j],zvs[k],0], X, Y, Z))**2).sum()