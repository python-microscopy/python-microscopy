#!/usr/bin/python
##################
# multicolourMeta.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from PYME.misc.djangoRecarray import qsToRecarray
from PYME.SampleDB import populateStats
from PYME.SampleDB.samples import models
from pylab import *

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
    figure(1)
    n, b = histogram(r['nEvents'], linspace(0, 4e5,50))
    errorbar((b[:-1] + b[1:])/2., n/float(len(r)), sqrt(n)/float(len(r)), lw=2, label=chan)#, where='post')
    #print n.max()
    xlabel('# Events')
    ylabel('Frequency')
    legend()

    figure(2)
    n, b = histogram(r['meanPhotons'], linspace(0, 6e3, 30))
    errorbar((b[:-1] + b[1:])/2., n/float(len(r)), sqrt(n)/float(len(r)), lw=2, label=chan)#, where='pre')
    xlabel('Mean Photon #')
    ylabel('Frequency')
    legend()

    figure(5)
    plot(r['meanPhotons'], r['nEvents'], '.', ms=3, label=chan)
    #print n.max()
    ylabel('Num Events')
    xlabel('Mean Photon #')
    xlim(0, 4e3)
    ylim(0, 4e5)
    #axis('equal')
    #ylim(0, 4e5)
    #xlim(0, (7./6)*4e5)
    #gca().set_aspect('equal')
    #title('Number of Events')
    legend()

    figure(6)
    n, b = histogram(r['tMedian'], linspace(0, 1e3, 20))
    step((b[:-1] + b[1:])/2., n/float(len(r)), sqrt(n)/float(len(r)), lw=2, label=chan, where='mid')#, where='pre')
    xlabel('Median decay time')
    ylabel('Frequency')
    legend()

    figure(7)
    plot(r['tMedian'], r['nEvents'], '.', ms=3, label=chan)
    #print n.max()
    ylabel('Num Events')
    xlabel('Median decay time')
    #xlim(0, 4e3)
    #ylim(0, 4e5)
    #axis('equal')
    #ylim(0, 4e5)
    #xlim(0, (7./6)*4e5)
    #gca().set_aspect('equal')
    #title('Number of Events')
    legend()

def drawPairwiseGraphs(chan1, chan2):
    r1, r2 = getChannelPair(chan1, chan2)
    print len(r1), len(r2)
    figure(3)
    plot(r1['nEvents'], r2['nEvents'], '.', ms=3, label=chan2)
    #print n.max()
    ylabel('A647 / A680')
    xlabel(chan1)
    #axis('equal')
    ylim(0, 4e5)
    xlim(0, (7./6)*4e5)
    gca().set_aspect('equal')
    title('Number of Events')
    legend()

    figure(4)
    plot(r1['meanPhotons'], r2['meanPhotons'], '.', ms=3, label=chan2)
    #print n.max()
    xlabel(chan1)
    ylabel('A647 / A680')
    ylim(0, 4e3)
    xlim(0, (7./6)*4e3)
    gca().set_aspect('equal')
    title('Mean Photon Number')
    legend()

def drawGraphs():
    close('all')
    for c in CHANNELS:
        drawChannelGraphs(c)

    for cp in PAIRS:
        drawPairwiseGraphs(*cp)




