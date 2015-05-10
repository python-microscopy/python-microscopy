#!/usr/bin/python

###############
# plotTimings.py
#
# Copyright David Baddeley, 2012
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
################

from pylab import *

def PlotTimings():
    f = open('poll.txt')
    
    events = {'q':[], 'p':[], 'b':[], 't':[], 'n':[]}
    pollEvents = {'b':[], 'n':[], 't':[]} 
    
    for line in f.readlines():
        time, event = line.strip().split('\t')
        time = float(time)
        
        events[event].append(time)
        
        if event == 'p':
            pollTime = time
        elif event in ['b', 't', 'n']:
            pollEvents[event].append([pollTime, time])
        
    f.close()
        
    figure()
    
    for i, ev in enumerate(events.keys()):
        t = array(events[ev])
        
        plot(t, i*ones(t.size), '+', label=ev)
        
    legend()
    axis('equal')
    
    figure()
    colours = ['r', 'g', 'b']
    for i, ev in enumerate(pollEvents.keys()):
        if len(pollEvents[ev]) > 0:
            t = array(pollEvents[ev])
            print((t.shape))
            
            u = arange(t.shape[0]) % 2
            
            hlines(ones(t.shape[0]) + .02*(u- .5), t[:,0], t[:,1], colours[i], label=ev, lw=20*(3-i))
    
    t = array(events['q'])
    vlines(t, .5*ones(t.size), 1.5*ones(t.size), label='q') 
    plot(t, 1.5*ones(t.size) +.1*rand(t.size), '+k')#, label='q')      
    legend()
    
    ylim(-1, 3)
    #axis('equal')