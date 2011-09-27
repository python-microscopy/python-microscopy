#!/usr/bin/python
##################
# filterPlots.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from pylab import *

def plotFilterMatrix(r, t, tp, rat, filterNames, ms = 20, dy=.4):
    figure()

    rs = array(list(set(r)))
    r_i = zeros(r.max()+1)
    r_i[rs] = arange(rs.size)

    ts = array(list(set(t)))
    t_i = zeros(t.max()+1)
    t_i[ts] = arange(ts.size)

    afn = array(filterNames)

    scatter(r_i[r], t_i[t], c = tp, s = ms, cmap=cm.gray_r, vmin=0, vmax=1)
    scatter(r_i[r], t_i[t]+dy, c = rat, s = ms, cmap=cm.gray_r, vmin=.2, vmax=.5)
    scatter(r_i[r]+dy/2, t_i[t]+dy/2, c = rat*tp, s = ms, cmap=cm.gray_r, vmin=.22, vmax=.25)

    a = gca()
    xticks(arange(rs.size))
    yticks(arange(ts.size))

    a.set_xticklabels(afn[rs], rotation='vertical')
    a.set_yticklabels(afn[ts])
    #grid()
    draw()

class plotFilterScatter(object):
    def __init__(self, throughput, ratios, minThroughput, minRatioDist, d_i, t_i, r_i, dichroicNames, filterNames):
        self.throughput = throughput
        self.ratios = ratios
        self.d_i = d_i
        self.t_i = t_i
        self.r_i = r_i
        self.dichroicNames = dichroicNames
        self.filterNames = filterNames

        self.f = figure()

        scatter(minThroughput, minRatioDist, c = rand(len(dichroicNames))[d_i], s = 10, linewidth=0, picker=50)
        self.f.canvas.mpl_connect('pick_event', self.onpick)

        self.t = self.f.text(.7, .85, '', bbox=dict(edgecolor='k', facecolor='w'))


    def onpick(self,event):
        n = event.ind
        d, t, r = self.d_i[n], self.t_i[n], self.r_i[n]
        s = '%s, %s, %s' % (self.dichroicNames[d], self.filterNames[t], self.filterNames[r])

        s += '\n%s\n%s' % (self.ratios[:, d,t,r].ravel(), self.throughput[:,d,t,r].ravel())

        #f.text(.8, .9, s, bbox=dict(edgecolor='k', facecolor='w'))
        self.t.set_text(s)
        draw()

        #figure(2)
        





