#!/usr/bin/python

##################
# tcHist.py
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

#from pylab import *

import numpy as np
# import pylab
import matplotlib.cm


def doTCHist(xvals, yvals, xbins, ybins, sat=1):
    h = np.histogram2d(xvals,yvals,[xbins,ybins])[0]
    lh = np.log10(h + 1).T
    #print lh.shape

    X,Y = np.meshgrid(xbins[:-1], ybins[:-1])

    c = matplotlib.cm.RdYlGn(np.minimum(np.maximum(X/(X + Y), 0),1))

    #print c.shape

    sc = np.minimum(sat*lh/lh.max(), 1)

    r = c[:,:,:3]
    r[:,:,0] = r[:,:,0]*sc
    r[:,:,1] = r[:,:,1]*sc
    r[:,:,2] = r[:,:,2]*sc

    return r

def doInvTCHist(xvals, yvals, xbins, ybins, sat=1):
    h = np.histogram2d(xvals,yvals,[xbins,ybins])[0]
    lh = np.log10(h + 1).T
    #print lh.shape

    X,Y = np.meshgrid(xbins[:-1], ybins[:-1])

    c = 1 - matplotlib.cm.RdYlGn(np.minimum(np.maximum(X/(X + Y), 0),1))

    #print c.shape

    sc = np.minimum(sat*lh/lh.max(), 1)

    r = c[:,:,:3]
    r[:,:,0] = r[:,:,0]*sc
    r[:,:,1] = r[:,:,1]*sc
    r[:,:,2] = r[:,:,2]*sc

    return 1-r

    
