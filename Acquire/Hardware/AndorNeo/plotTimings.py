# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:49:19 2011

@author: dbad004
"""
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
            print t.shape
            
            u = arange(t.shape[0]) % 2
            
            hlines(ones(t.shape[0]) + .02*(u- .5), t[:,0], t[:,1], colours[i], label=ev, lw=20*(3-i))
    
    t = array(events['q'])
    vlines(t, .5*ones(t.size), 1.5*ones(t.size), label='q') 
    plot(t, 1.5*ones(t.size) +.1*rand(t.size), '+k')#, label='q')      
    legend()
    
    ylim(-1, 3)
    #axis('equal')