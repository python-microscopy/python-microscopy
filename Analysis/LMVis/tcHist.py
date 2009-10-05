#!/usr/bin/python

##################
# tcHist.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from pylab import *
import scipy as sp


def doTCHist(xvals, yvals, xbins, ybins, sat=1):
    h = sp.histogram2d(xvals,yvals,[xbins,ybins])[0]
    lh = log10(h + 1).T
    #print lh.shape

    X,Y = sp.meshgrid(xbins[:-1], ybins[:-1])

    c = cm.RdYlGn(sp.minimum(sp.maximum(X/(X + Y), 0),1))

    #print c.shape

    sc = sp.minimum(sat*lh/lh.max(), 1)

    r = c[:,:,:3]
    r[:,:,0] = r[:,:,0]*sc
    r[:,:,1] = r[:,:,1]*sc
    r[:,:,2] = r[:,:,2]*sc

    return r

def doInvTCHist(xvals, yvals, xbins, ybins, sat=1):
    h = sp.histogram2d(xvals,yvals,[xbins,ybins])[0]
    lh = log10(h + 1).T
    #print lh.shape

    X,Y = sp.meshgrid(xbins[:-1], ybins[:-1])

    c = 1 - cm.RdYlGn(sp.minimum(sp.maximum(X/(X + Y), 0),1))

    #print c.shape

    sc = sp.minimum(sat*lh/lh.max(), 1)

    r = c[:,:,:3]
    r[:,:,0] = r[:,:,0]*sc
    r[:,:,1] = r[:,:,1]*sc
    r[:,:,2] = r[:,:,2]*sc

    return 1-r

    
