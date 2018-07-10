#!/usr/bin/python

##################
# piezo_test.py
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

import scipy
import matplotlib.pyplot as plt
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
    plt.plot(data[0], data[1] - data[0])

def save_data(data, fname):
    s = ''
    for i in range(len(data[0])):
        for j in range(len(data)):
            s = s + '%s ' % (data[j][i],)
        s = s + '\n'

    f = open(fname, 'w')
    f.write(s)
    f.close()
