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

def PlotShiftField(dx, dy, spx, spy):
    figure()
    #subplot(221)
    axes([.05, .7, .4, .25])
    imshow(dx.T)
    axis('image')
    axes([.55, .7, .4, .25])
    imshow(dy.T)
    axis('image')

    axes([.05, .05, .9, .6])
    xin, yin = meshgrid(arange(0, 512*70, 500), arange(0, 256*70, 1000))
    xin = xin.ravel()
    yin = yin.ravel()
    quiver(xin, yin, spx.ev(xin, yin), spy.ev(xin, yin), scale=1e4)
    axis('image')