# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:56:18 2015

@author: david
"""
import numpy as np
import toposort
import networkx as nx
import pylab

def makeGraph(dg):
    G = nx.DiGraph()
    for k, v in dg.items():
        for e in v:
            G.add_edge(e, k)
    
    return G

def arrangeNodes(dg):
    ts = list(toposort.toposort(dg))
    xc = 0
    ips = {}
    for st in ts:
        yps = [];
        st = list(st)
        for si in st:
            if si in dg.keys():
                ri = list(dg[si])
                yp = np.mean([ips[rr][1] for rr in ri])
            else:
                yp = 0
            
            while yp in yps:
               yp += 1
               
            yps.append(yp)
                
        #ysi = np.argsort(yps) 
        #ys = (np.arange(len(ysi)) - ysi.mean())
        ysi = np.arange(len(yps))
        ys = yps
        for i, yi in zip(ysi, ys):
            ips[st[i]] = (xc, yi)
        
        xc += 1
        
    return ips
    
def drawGraph(dg):
    ips = arrangeNodes(dg)
    #ts = list(toposort.toposort(dg))
    
    f = pylab.figure()
    a = pylab.axes([0,0,1,1])
    
    cols = {}    
    for k, v in dg.items():
        if not isinstance(k, str) or isinstance(k, unicode):
            yv0 = []
            yoff = .1*np.arange(len(v))
            yoff -= yoff.mean()
            
            for e in v:
                x0, y0 = ips[e]
                yv0.append(y0 + 0.01*x0)
                
            yvi = np.argsort(np.array(yv0))
            #print yv0, yvi
            yos = np.zeros(3)
            yos[yvi] = yoff
                
            for e, yo in zip(v, yos):
                x0, y0 = ips[e]
                x1, y1 = ips[k]
                
                if not e in cols.keys():
                    cols[e] = 0.7*np.array(pylab.cm.hsv(pylab.rand()))

                #yo = yoff[i]                
                
                pylab.plot([x0,x0+.5, x0+.5, x1], [y0,y0,y1+yo,y1+yo], c=cols[e], lw=2)
                
    for k, v in ips.items():   
        if not isinstance(k, str) or isinstance(k, unicode):
            s = k.__class__.__name__
            #pylab.plot(v[0], v[1], 'o', ms=5)
            rect = pylab.Rectangle([v[0], v[1]-.25], 1, .5, ec='k', fc=[.8,.8, 1], picker=True)
            
            rect._data = k
            pylab.gca().add_patch(rect)
            pylab.text(v[0]+.05, v[1]+.18 , s, weight='bold')
            
            s2 = '\n'.join(['%s : %s' %i for i in k.get().items()])
            pylab.text(v[0]+.05, v[1]-.22 , s2, size=8, stretch='ultra-condensed')
        else:
            s = k
            if not k in cols.keys():
                cols[k] = 0.7*np.array(pylab.cm.hsv(pylab.rand()))
            pylab.plot(v[0], v[1], 'o', color=cols[k])
            pylab.text(v[0]+.1, v[1] + .02, s, color=cols[k], weight='bold')
                
                
    #pylab.ylim(-1, 2)
                
    ipsv = np.array(ips.values())
    xmn, ymn = ipsv.min(0)
    xmx, ymx = ipsv.max(0)
    
    pylab.ylim(ymn-1, ymx+1)
    pylab.xlim(xmn-.5, xmx + .7)
    
    pylab.axis('off')
    
    def OnPick(event):
        k = event.artist._data
        if not isinstance(k, str) or isinstance(k, unicode):
            k.edit_traits()
        
    f.canvas.mpl_connect('pick_event', OnPick)
        