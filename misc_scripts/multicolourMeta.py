#!/usr/bin/python
##################
# multicolourMeta.py
#
# Copyright David Baddeley, 2010
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

"""
Extract dye infomation / statistics from sample database
"""

from PYME.misc.djangoRecarray import qsToRecarray
from SampleDB2 import populateStats
from SampleDB2.samples import models
import numpy as np
import matplotlib.pyplot as plt
#from pylab import *

CHANNELS = [ 'A680', 'A647','A750',]
PAIRS = [('A750', 'A647'),('A750', 'A680'), ]

def getChannel(chan):
    return qsToRecarray(models.EventStats.objects.filter(label=chan))

def getChannelPair(chan1, chan2):
    files = models.File.objects.filter(event_stats__label=chan1).filter(event_stats__label=chan2)
    r1 = qsToRecarray(models.EventStats.objects.filter(fileID__in=files, label=chan1))
    r2 = qsToRecarray(models.EventStats.objects.filter(fileID__in=files, label=chan2))

    return r1, r2

def drawChannelGraphs(chan):
    r = getChannel(chan)
    plt.figure(1)
    n, b = np.histogram(r['nEvents'], np.linspace(0, 4e5,50))
    plt.errorbar((b[:-1] + b[1:])/2., n/float(len(r)), np.sqrt(n)/float(len(r)), lw=2, label=chan)#, where='post')
    #print n.max()
    plt.xlabel('# Events')
    plt.ylabel('Frequency')
    plt.legend()

    plt.figure(2)
    n, b = np.histogram(r['meanPhotons'], np.linspace(0, 6e3, 30))
    #errorbar((b[:-1] + b[1:])/2., n/float(len(r)), sqrt(n)/float(len(r)), lw=2, label=chan)#, where='pre')
    plt.plot((b[:-1] + b[1:])/2., n/float(len(r)), lw=2, label=chan)
    plt.xlabel('Mean Photon #')
    plt.ylabel('Frequency')
    plt.legend()

    plt.figure(5)
    plt.plot(r['meanPhotons'], r['nEvents'], '.', ms=3, label=chan)
    #print n.max()
    plt.ylabel('Num Events')
    plt.xlabel('Mean Photon #')
    plt.xlim(0, 4e3)
    plt.ylim(0, 4e5)
    #axis('equal')
    #ylim(0, 4e5)
    #xlim(0, (7./6)*4e5)
    #gca().set_aspect('equal')
    #title('Number of Events')
    plt.legend()

    plt.figure(6)
    n, b = np.histogram(r['tMedian'], np.linspace(0, 1e3, 20))
    plt.step((b[:-1] + b[1:])/2., n/float(len(r)), lw=2, label=chan, where='mid')#, where='pre')
    plt.xlabel('Median decay time')
    plt.ylabel('Frequency')
    plt.legend()

    plt.figure(7)
    plt.plot(r['tMedian'], r['nEvents'], '.', ms=3, label=chan)
    #print n.max()
    plt.ylabel('Num Events')
    plt.xlabel('Median decay time')
    #xlim(0, 4e3)
    #ylim(0, 4e5)
    #axis('equal')
    #ylim(0, 4e5)
    #xlim(0, (7./6)*4e5)
    #gca().set_aspect('equal')
    #title('Number of Events')
    plt.legend()

def drawPairwiseGraphs(chan1, chan2):
    r1, r2 = getChannelPair(chan1, chan2)
    print((len(r1), len(r2)))
    plt.figure(3)
    plt.plot(r1['nEvents'], r2['nEvents'], '.', ms=3, label=chan2)
    #print n.max()
    plt.ylabel('A647 / A680')
    plt.xlabel(chan1)
    #axis('equal')
    plt.ylim(0, 4e5)
    plt.xlim(0, (7./6)*4e5)
    plt.gca().set_aspect('equal')
    plt.title('Number of Events')
    plt.legend()

    plt.figure(4)
    plt.plot(r1['meanPhotons'], r2['meanPhotons'], '.', ms=3, label=chan2)
    #print n.max()
    plt.xlabel(chan1)
    plt.ylabel('A647 / A680')
    plt.ylim(0, 4e3)
    plt.xlim(0, (7./6)*4e3)
    plt.gca().set_aspect('equal')
    plt.title('Mean Photon Number')
    plt.legend()

def drawGraphs():
    plt.close('all')
    #for c in CHANNELS:
    #    drawChannelGraphs(c)

    #for cp in PAIRS:
    #    drawPairwiseGraphs(*cp)

    plt.figure(8)
    N = [getChannel(c)['meanPhotons'] for c in CHANNELS]
    plt.hist(N, np.linspace(0, 6e3, 30), normed=1)
    plt.xlabel('Mean Photon #')
    plt.ylabel('Frequency')
    #legend()




