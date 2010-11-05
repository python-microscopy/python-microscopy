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
    colorbar()
    axes([.55, .7, .4, .25])
    imshow(dy.T)
    axis('image')
    colorbar()

    axes([.05, .05, .9, .6])
    xin, yin = meshgrid(arange(0, 512*70, 500), arange(0, 256*70, 1000))
    xin = xin.ravel()
    yin = yin.ravel()
    quiver(xin, yin, spx.ev(xin, yin), spy.ev(xin, yin), scale=1e4)
    axis('image')


def PlotShiftField2(spx, spy):
    xi, yi = meshgrid(arange(0, 512*70, 100), arange(0, 256*70, 100));xin = xi.ravel();yin = yi.ravel()
    dx = spx.ev(xin[:], yin[:]).reshape(xi.shape)
    dy = spy.ev(xin[:], yin[:]).reshape(xi.shape)
    figure()
    subplot(311)
    #axes([.05, .7, .4, .25])
    imshow(dx[::-1, :], extent=[0,512, 0, 256], clim=[-70, 20])
    axis('image')
    colorbar()
    #axes([.55, .7, .4, .25])
    subplot(312)
    im = imshow(dy[::-1, :], extent=[0,512, 0, 256], clim=[-70, 20])
    axis('image')
    colorbar()

    #axes([.05, .05, .9, .6])
    xi, yi = meshgrid(arange(0, 512*70, 2200), arange(0, 256*70, 2200));xin = xi.ravel();yin = yi.ravel()
    dx = spx.ev(xin[:], yin[:]).reshape(xi.shape)
    dy = spy.ev(xin[:], yin[:]).reshape(xi.shape)
    subplot(313)
    
    quiver(xin/70, yin/70, spx.ev(xin, yin), spy.ev(xin, yin), scale=1e3)
    
    axis('image')
    xlim(0, 512)
    ylim(0, 256)

    colorbar(im)