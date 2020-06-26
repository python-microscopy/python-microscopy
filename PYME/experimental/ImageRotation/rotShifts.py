#!/usr/bin/python

##################
# rotShifts.py
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
# from pylab import *
from matplotlib.pyplot import *

def rotMod(p, x1,y1):
    dx,dy,xc, yc, theta = p
    #dx = 0
    #dy = 0
    
    sx,sy = rotVects(x1,y1,xc,yc,theta)

    #plot(hstack((sx + dx, sy + dy)))
    quiver(x1, y1, sx+dx, sy+dy)
    return hstack((sx + dx, sy + dy))


def rotVects(x1,y1, xc, yc, theta):
    dx = xc - x1
    dy = yc - y1
    sx = 2*sin(theta/2)*(sin(theta/2)*dx - cos(theta/2)*dy)
    sy = 2*sin(theta/2)*(cos(theta/2)*dx + sin(theta/2)*dy)
    return (sx, sy)


def rotVects_c(x1,y1, xc, yc, theta):
    dx = xc - x1
    dy = yc - y1
    sx = 2*sin(theta/2)*(sin(theta/2)*dx - cos(theta/2)*dy)
    sy = 2*sin(theta/2)*(cos(theta/2)*dx + sin(theta/2)*dy)
    return (sx + xc, sy + yc)
