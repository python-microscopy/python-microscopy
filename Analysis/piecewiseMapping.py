#!/usr/bin/python

##################
# piecewiseMapping.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from numpy import *
import sys

def timeToFrames(t, events, mdh):
    cycTime = mdh.getEntry('Camera.CycleTime')
    startTime = mdh.getEntry('StartTime')

    se = array([('0', 'start', startTime)], dtype=events.dtype)
    sf = array([('%d' % sys.maxint, 'start', 1e100)], dtype=events.dtype)
    #get events corresponding to aquisition starts
    startEvents = hstack((se, events[events['EventName'] == 'StartAq'], sf))

    sfr = array([int(e['EventDescr']) for e in startEvents])

    si = startEvents['Time'].searchsorted(t, side='right')
    fr = minimum(sfr[si-1] + floor((t - startEvents['Time'][si-1]) / cycTime), sfr[i])

    return fr

def framesToTime(fr, events, mdh):
    cycTime = mdh.getEntry('Camera.CycleTime')
    startTime = mdh.getEntry('StartTime')

    se = array([('0', 'start', startTime)], dtype=events.dtype)
    #get events corresponding to aquisition starts
    startEvents = hstack((se, events[events['EventName'] == 'StartAq']))

    sfr = array([int(e['EventDescr']) for e in startEvents])

    si = sfr.searchsorted(fr, side = 'right')
    return startEvents['Time'][si-1] + (fr - sfr[si-1]) * cycTime
    

class piecewiseMap:
    def __init__(self, y0, xvals, yvals, secsPerFrame=1, xIsSecs=True):
        self.y0 = y0

        if xIsSecs: #store in frame numbers
            self.xvals = xvals / secsPerFrame
        else:
            self.xvals = xvals
        self.yvals = yvals

        self.secsPerFrame = secsPerFrame
        self.xIsSecs = xIsSecs

    def __call__(self, xp, xpInFrames=True):
        yp = 0 * xp

        if not xpInFrames:
            xp = xp / self.secsPerFrame
        
#        y0 = self.y0
#        x0 = -inf
#
#        for x, y in zip(self.xvals, self.yvals):
#            yp += y0 * (xp >= x0) * (xp < x)
#            x0, y0 = x, y
#
#        x  = +inf
#        yp += y0 * (xp >= x0) * (xp < x)

        inds = self.xvals.searchsorted(xp)
        yp  = self.yvals[maximum(inds-1, 0)]
        yp[inds == 0] = self.y0

        return yp

def GeneratePMFromProtocolEvents(events, metadata, x0, y0, id='setPos', idPos = 1, dataPos=2):
    x = []
    y = []

    secsPerFrame = metadata.getEntry('Camera.CycleTime')

    for e in events[events['EventName'] == 'ProtocolTask']:
        #if e['EventName'] == eventName:
        ed = e['EventDescr'].split(', ')
        if ed[idPos] == id:
            x.append(e['Time'])
            y.append(float(ed[dataPos]))

    return piecewiseMap(y0, timeToFrames(array(x), events, metadata), array(y), secsPerFrame, xIsSecs=False)


def GeneratePMFromEventList(events, metadata, x0, y0, eventName='ProtocolFocus', dataPos=1):
    x = []
    y = []

    secsPerFrame = metadata.getEntry('Camera.CycleTime')

    for e in events[events['EventName'] == eventName]:
        #if e['EventName'] == eventName:
        x.append(e['Time'])
        y.append(float(e['EventDescr'].split(', ')[dataPos]))

    return piecewiseMap(y0, timeToFrames(array(x), events, metadata), array(y), secsPerFrame, xIsSecs=False)

def GenerateBacklashCorrPMFromEventList(events, metadata, x0, y0, eventName='ProtocolFocus', dataPos=1, backlash=0):
    x = []
    y = []

    secsPerFrame = metadata.getEntry('Camera.CycleTime')

    for e in events[events['EventName'] == eventName]:
        #if e['EventName'] == eventName:
        x.append(e['Time'])
        y.append(float(e['EventDescr'].split(', ')[dataPos]))

    x = array(x)
    y = array(y)

    dy = diff(hstack(([y0], y)))

    for i in range(1, len(dy)):
        if dy[i] == 0:
            dy[i] = dy[i-1]

    y += backlash*(dy < 0)


    return piecewiseMap(y0, timeToFrames(x, events, metadata), y, secsPerFrame, xIsSecs=False)

