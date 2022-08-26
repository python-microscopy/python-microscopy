# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:56:18 2015

@author: david
"""
import numpy as np
import toposort
# import pylab
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy import optimize
from six.moves import xrange
import six

def makeGraph(dg):
    import networkx as nx
    
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

                if len(ri) > 0:
                    #assign a y position as the mean of the dependancies y positions
                    #yp = np.mean([ips[rr][1] for rr in ri])
                    w = np.array([1.0/(1 + (xc - ips[rr][0])) for rr in ri])
                    yp = np.sum(np.array([ips[rr][1] for rr in ri])*w)/w.sum()
                    ypw.append(1.)
                else:
                    # module has no inputs (i.e. a simulation module)
                    yp = 0
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
                fd_ws = []
                
                if (isinstance(si,six.string_types)):
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
                                    fd_ws += [1.0/(1 + ips[rr][0] - xc) for rr in gnodes if rr in ips.keys()]
                
                if len(fd_ys) > 0:
                    ypf = np.mean(fd_ys)
                    #fd_ws = np.array(fd_ws)
                    #ypf = np.sum(np.array(fd_ys)*fd_ws)/np.sum(fd_ws)
                    
                    #print ypf
    
                    if ypw[-1] == 1:
                        yp = 0.5*yp + 0.5*ypf
                    else:
                        yp = ypf
            
            #print yp, ypf
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
            print('NaN detected in yest')
            yest = ypss
            
            if np.any(np.isnan(ypss)):
                print('NaN detected in ypss ??!!')
                
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


def _vertCost(yvs, vert_neighbours, ysize=1.0):
    #ynn = [yvs[n] for n in vert_neighbours]

    cost = []
    for y, n in zip(yvs, vert_neighbours):
        c = np.max(np.max(y - (yvs[n] - ysize)), 0)*np.max(np.max(-y + (yvs[n] + ysize)), 0)
        cost.append(c)

    return np.array(cost)

def _vertForce(yvs, vert_neighbours, ysize=1.0):
    #ynn = [yvs[n] for n in vert_neighbours]

    force = 0*yvs
    for i, y, n in zip(xrange(len(yvs)), yvs, vert_neighbours):
        if len(n) > 0:
            yn = yvs[n]
            #f1 = np.sum(np.max(y - (yn + ysize), 0)*(y < (yn + ysize)))
            #f2 = np.sum(np.max(-y + (yn - ysize), 0)*(y > (yn - ysize)))

            #ft = f1*f2
            #if (ft) > 0:
            #    c = (2.0*(f1 > f2) - 1)
            #    force[i] = c

            #force[i] = np.sum(1.0*(y>yn)*(y<(yn+ysize)) - 1.0*(y<=yn)*(y>(yn-ysize)))

            force[i] = np.sum(0.5*(1 + np.tanh((-abs(y-yn) + ysize)*5))*(2.0*(y>yn) - 1))

    return force

def __edgeCost(yvs, xvs, edges):
    dy = np.diff(yvs[edges],1)
    dx = np.diff(xvs[edges],1)

    return np.sqrt(dx*dx + dy*dy)

def _edgeForce(yvs, xvs, edges):
    force = 0*yvs
    for x, y, e, i in zip(xvs, yvs, edges, xrange(len(yvs))):
        dx = x - xvs[e]
        dy = y - yvs[e]

        r = np.sqrt(dx*dx + dy*dy)
        force[i] = np.sum(-dy/r*r)/np.sum(1.0/r)
        #mi = np.argmax(abs(dy)/r)
        #force[i] = -dy[mi]/r[mi]

    return force

def _totForce(yvs, xvs, edges, vert_neighbours, vsize=1.0, vert_weight=1.0):
    return vert_weight*_vertForce(yvs, vert_neighbours, ysize=1.0) + _edgeForce(yvs, xvs, edges)



def _totCost(yvs, xvs, edges, vert_neighbours, vsize=1.0):
    return _vertCost(yvs, vert_neighbours, ysize=1.0).sum() + _edgeCost(yvs, xvs, edges).sum()


def arrangeNodes_(dg):
    ts = list(toposort.toposort(dg))
    xc = 0
    ips = {}

    i = 0

    yvs = []
    xvs = []
    nodes = []

    vert_neighbours = []

    node_nums = {}

    edge_db = {}
    edges = []


    #ts gives a list of all the steps of the computation
    for step in ts:
        #iterate over the steps
        step = list(step)

        yis = []
        #loop over items to be calculated at each step
        #and work out a preferred position
        for si in step:
            xvs.append(xc)
            yvs.append(np.random.randn())
            nodes.append(si)

            node_nums[si] = i
            yis.append(i)

            #see what the dependancies of this item are
            #record the edges
            if si in dg.keys():
                ri = list(dg[si])
                for r in ri:
                    n = node_nums[r]
                    edges.append((n, i))

                    old_edges = edge_db.get(i, [])
                    edge_db[i] = old_edges + [n,]

                    old_edges = edge_db.get(n, [])
                    edge_db[n] = old_edges + [i, ]

            i += 1

        for yi in yis:
            vert_neighbours.append([yi_ for yi_ in yis if not yi_  == yi])

        #vert_neighbours.append(yis)

        xc += 1

    edgel = [edge_db[k] for k in range(i)]

    return np.array(xvs), np.array(yvs), nodes, vert_neighbours, edgel, edges



    
def drawGraph(dg):
    ips = arrangeNodes(dg)
    #ts = list(toposort.toposort(dg))
    
    f = plt.figure()
    a = plt.axes([0,0,1,1])
    
    axisWidth = a.get_window_extent().width
    nCols = max([v[0] for v in ips.values()])
    pix_per_col = axisWidth/float(nCols)
    
    fontSize = min(10, 10*pix_per_col/400.)
    
    #print pix_per_col, fontSize
    
    cols = {}    
    for k, v in dg.items():
        if not isinstance(k, six.string_types):
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
                    cols[e] = 0.7*np.array(matplotlib.cm.hsv(np.random.rand()))

                #yo = yoff[i]                
                
                plt.plot([x0,x0+.5, x0+.5, x1], [y0,y0,y1+yo,y1+yo], c=cols[e], lw=2)
                
    for k, v in ips.items():   
        if not isinstance(k, six.string_types):
            s = k.__class__.__name__
            #plt.plot(v[0], v[1], 'o', ms=5)
            rect = plt.Rectangle([v[0], v[1]-.25], 1, .5, ec='k', fc=[.8,.8, 1], picker=True)
            
            rect._data = k
            plt.gca().add_patch(rect)
            plt.text(v[0]+.05, v[1]+.18 , s, size=fontSize, weight='bold')
            
            s2 = '\n'.join(['%s : %s' %i for i in k.get().items()])
            plt.text(v[0]+.05, v[1]-.22 , s2, size=.8*fontSize, stretch='ultra-condensed')
        else:
            s = k
            if not k in cols.keys():
                cols[k] = 0.7*np.array(matplotlib.cm.hsv(np.random.rand()))
            plt.plot(v[0], v[1], 'o', color=cols[k])
            plt.text(v[0]+.1, v[1] + .02, s, color=cols[k], size=fontSize, weight='bold')
                
                
    #plt.ylim(-1, 2)
                
    ipsv = np.array(ips.values())
    xmn, ymn = ipsv.min(0)
    xmx, ymx = ipsv.max(0)
    
    plt.ylim(ymn-1, ymx+1)
    plt.xlim(xmn-.5, xmx + .7)
    
    plt.axis('off')
    
    def OnPick(event):
        k = event.artist._data
        if not isinstance(k, six.string_types):
            k.edit_traits()
        
    f.canvas.mpl_connect('pick_event', OnPick)
        