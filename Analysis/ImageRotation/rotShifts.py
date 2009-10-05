#!/usr/bin/python

##################
# rotShifts.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from numpy import *
from pylab import *

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
