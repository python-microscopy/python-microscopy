#!/usr/bin/python

##################
# twoColour.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from pylab import *

def Plot

    xin, yin = meshgrid(arange(0, 512*70, 500), arange(0, 256*70, 1000))
    clf()
    xin = xin.ravel()
    yin = yin.ravel()
    quiver(xin, yin, ix.ev(xin, yin), i.ev(xin, yin), scale=1e4)
    axis('image')