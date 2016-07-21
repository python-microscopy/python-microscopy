# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:56:18 2015

@author: david
"""
import numpy as np
import toposort
import networkx as nx
import pylab
from scipy import optimize

def makeGraph(dg):
    G = nx.DiGraph()
    for k, v in dg.items():
        for e in v:
            G.add_edge(e, k)
    
    return G
    
def _pos_cost(y_s, yps, ypw, tol=1.0):
    #print yps.shape, y_s.shape, 
    return np.sum(ypw*((-9 + np.cumsum(y_s**2) + tol*np.arange(len(y_s))) - yps)**2)
    
def _pos_cost_ls(y_s, yps, ypw, tol=1.0):
    #print yps.shape, y_s.shape, 
    return ypw*((-9 + np.cumsum(y_s**2) + tol*np.arange(len(y_s))) - yps)

def arrangeNodes(dg):
    ts = list(toposort.toposort(dg))
    xc = 0
    ips = {}
    
    yvs = []

    forward_deps = {}
    for k, v in dg.items():
        for vi in v:
            try: 
                forward_deps[vi].add(k)
            except KeyError:
                forward_deps[vi] = {k}
    
    
    #ts gives a list of all the steps of the computation
    for st in ts:
        #iterate over the steps

        #keep a list of y positions at this step
        yps = []
        ypfs = []
        ypw = []
        
        st = list(st)
        
        #loop over items to be calculated at each step
        #and work out a preferred position
        for si in st: 
            #see what the dependancies of this item are
            if si in dg.keys():
                ri = list(dg[si])
                
                #assign a y position as the mean of the dependancies y positions
                yp = np.mean([ips[rr][1] for rr in ri])
                ypw.append(1.)
            else:
                #else assign a position of 0
                yp = 0
                ypw.append(0.1)                
                
            #yps.append(yp)
            
            ypf = 0
            
            #look for forward dependencies
            if si in forward_deps.keys():
                outputs = forward_deps[si]
                fd_ys = []
                
                if (isinstance(si, str) or isinstance(si, unicode)):
                    pass
                    #we are a result node - look 1 step ahead
                    
                    #dependencies of forward deps
                    #for fdi in fd:
                    #    fdd = list(dg[fdi])
                        
                    #    fd_ys += [ips[rr][1] for rr in fdd if rr in ips.keys()]
                else:
                    #We are a computation node - look 2 steps ahead and backwards
                    
                    #look over the node outputs
                    for out_i in outputs:
                        #find the nodes which consume these outputs
                        if out_i in forward_deps.keys():
                            consuming_nodes = forward_deps[out_i]
                            
                            for cnode in consuming_nodes:
                                #find the nodes on which these nodes depend (these will be other outputs)
                                cnode_inputs = list(dg[cnode])
                            
                                for inp_i in cnode_inputs:
                                    #find the computational nodes which generate these outputs
                                    try:
                                        gnodes = list(dg[inp_i])
                                    except KeyError:
                                        gnodes = []
                            
                                    fd_ys += [ips[rr][1] for rr in gnodes if rr in ips.keys()]
                
                if len(fd_ys) > 0:                
                    ypf = np.mean(fd_ys)
                    
                    #print ypf
    
                    if ypw[-1] == 1:
                        yp = 0.5*yp + 0.5*ypf
                    else:
                        yp = ypf
            
            print yp, ypf            
            yps.append(yp)
            ypfs.append(ypf)
            
        
        Is = np.argsort(yps)
        
        #space out positions
        ypss = np.zeros(len(yps)) - 50       
        for i in np.argsort(yps):
            si = st[i]
            yp = yps[i]
            if isinstance(si, str):
                #vertical spacing between outputs is .1
                tol = .1
            else:
                #vertical spacing between blocks
                tol = 1
            
            #space out the y positions so blocks don't overlap
            #while min(abs(yp - np.array(ypss + [-50]))) < tol:
            while min(abs(yp - ypss)) < tol:
               yp += min(tol, .1)
               
            ypss[i] = yp
        
        yest = ypss.copy()
        #print ypss
        
        if len(yest) > 1:
            sp = 0*yest
            sp[1:] = np.diff(yest[Is] - tol*np.arange(len(yps))) 
            sp[0] = yest[Is][0]+9
            sp = np.sqrt(sp)
            #yest[Is] = -9 + np.cumsum(optimize.fmin(_pos_cost, sp, (np.array(yps)[Is], np.array(ypw)[Is], tol), disp=1)[0]**2) + tol*np.arange(len(yps))
            yest[Is] = -9 + np.cumsum(optimize.leastsq(_pos_cost_ls, sp, (np.array(yps)[Is], np.array(ypw)[Is], tol))[0]**2) + tol*np.arange(len(yps))
        
        #print yps, yest, tol
        
        
        if np.any(np.isnan(yest)):
            print 'NaN detected in yest'
            yest = ypss
            
            if np.any(np.isnan(ypss)):
                print 'NaN detected in ypss ??!!'
                
        #ysi = np.argsort(yps) 
        #ys = (np.arange(len(ysi)) - ysi.mean())
                
        #assign the y positions
        ysi = np.arange(len(ypss))
        #ys = ypss
        #for i, yi in zip(ysi, ys):
        #    ips[st[i]] = (xc, yi)
            
        for i, yi in zip(ysi, yest):
            ips[st[i]] = (xc, yi)
        
        xc += 1
        
        yvs.append(yest[np.argsort(yest)])
        
    return ips, yvs
    
def drawGraph(dg):
    ips = arrangeNodes(dg)
    #ts = list(toposort.toposort(dg))
    
    f = pylab.figure()
    a = pylab.axes([0,0,1,1])
    
    axisWidth = a.get_window_extent().width
    nCols = max([v[0] for v in ips.values()])
    pix_per_col = axisWidth/float(nCols)
    
    fontSize = min(10, 10*pix_per_col/400.)
    
    print pix_per_col, fontSize
    
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
            pylab.text(v[0]+.05, v[1]+.18 , s, size=fontSize, weight='bold')
            
            s2 = '\n'.join(['%s : %s' %i for i in k.get().items()])
            pylab.text(v[0]+.05, v[1]-.22 , s2, size=.8*fontSize, stretch='ultra-condensed')
        else:
            s = k
            if not k in cols.keys():
                cols[k] = 0.7*np.array(pylab.cm.hsv(pylab.rand()))
            pylab.plot(v[0], v[1], 'o', color=cols[k])
            pylab.text(v[0]+.1, v[1] + .02, s, color=cols[k], size=fontSize, weight='bold')
                
                
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
        