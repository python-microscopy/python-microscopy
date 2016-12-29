#!/usr/bin/python

##################
# shiftDisp.py
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




def get_r():
    if tq.getNumberTasksCompleted() > 0:
        t = tq.getCompletedTask()
        xr = []
        yr = []
        xg = []
        yg = []
        for i in range(len(t.results[0])):
            r_r = t.results[0][i]
            r_g = t.results[1][i]
            if not r_r.fitErr is None and not r_g.fitErr is None and r_r.fitErr[1] < 50 and r_g.fitErr[1] < 50:
                xr.append(r_r.x0())
                yr.append(r_r.y0())
                xg.append(r_g.x0())
                yg.append(r_g.y0())
        xr = array(xr)
        yr = array(yr)
        xg = array(xg)
        yg = array(yg)
        clf()
        quiver(xr, yr, (xr-xg), yr-yg)
