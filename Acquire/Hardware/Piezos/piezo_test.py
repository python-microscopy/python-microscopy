#!/usr/bin/python

##################
# piezo_test.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import scipy
import scipy.plt
import time

def test_piezo(piezo, startpos=0, endpos=0, stepsize=0.04, chan = 1, delay=0):
    if endpos == 0:
        endpos = piezo.GetMax(chan)

    pos = scipy.arange(startpos, endpos, stepsize)
    a = scipy.zeros(len(pos), 'd')

    for i in range(len(pos)):
        piezo.MoveTo(chan,pos[i], 0)
        if delay > 0:
            time.sleep(delay)
        a[i] = (piezo.GetPos(chan))

    return (pos, a)

def plot_diff(data):
    scipy.plt.plot(data[0], data[1] - data[0])

def save_data(data, fname):
    s = ''
    for i in range(len(data[0])):
        for j in range(len(data)):
            s = s + '%s ' % (data[j][i],)
        s = s + '\n'

    f = file(fname, 'w')
    f.write(s)
    f.close()
