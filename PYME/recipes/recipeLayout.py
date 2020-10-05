import numpy as np
from PYME.misc import depGraph
from scipy import optimize
import six

def _path_lh(y, yprev, ygoal, yvs, ytol):
    d_yprev = (y - yprev) ** 2
    d_ygoal = (y - ygoal) ** 2
    d_obstructions = (((y - yvs) / ytol) ** 2).min()

    return 10 * d_yprev * (d_yprev < .3 ** 2) + 0 * d_ygoal - 10 * d_obstructions * (d_obstructions < 1)


def _path_v_lh(x, xtarget, y, vlines, vdir):
    d_x = np.array([((x - vl[0, 0]) / 0.1) ** 2 for vl in vlines])

    #odir = np.array([np.sign(vl[1,1] - vl[1,0])])
    #d_yprev = (y - yprev)**2
    #d_ygoal = (y-ygoal)**2
    #d_obstructions = (((y-yvs)/ytol)**2).min()

    return -10 * np.sum(d_x) * (np.sum(d_x < 1) > 0)


def layout(dg):
    """
    Find the input lines coming in to each processing node

    Parameters
    ----------
    dg : the dependency graph

    Returns
    -------

    """
    ips, yvs = depGraph.arrangeNodes(dg)
    yvs = [list(yi) for yi in yvs]
    yvs_e = [{} for yi in yvs]
    yvtol = [list(0 * np.array(yi) + .35) for yi in yvs]
    line_x_offsets = {}

    vlines = [[] for i in range(len(yvs))]

    connecting_lines = []
    for k, deps in dg.items():
        if not (isinstance(k, six.string_types)):
            #This is a processing node
            yv0 = []

            #deps = list(deps)

            #each input line should be offset / spaced from the others
            yoff = .1 * np.arange(len(deps))
            yoff -= yoff.mean()

            ##########
            #loop over the inputs and determine their y-order
            for e in deps:
                x0, y0 = ips[e] #position of start of lines
                x1, y1 = ips[k] #nominal (non-offset) end points

                #if y0>y1, offset y0 slightly +ve, else offset slightly negative
                yv0.append(y0 + 0.01 * (x1 - x0) * (2.0 * (y0 > y1) - 1))

            #sort in ascending order of y0 values
            yvi = np.argsort(np.array(yv0))
            #print yv0, yvi

            ##########
            #assign the correct y offset values
            #to each line
            yos = np.zeros(len(yvi))
            yos[yvi] = yoff

            ##########
            # Plot the lines
            for e, yo in zip(deps, yos):
                x0, y0 = ips[e] #start pos of input line
                x1, y1 = ips[k] #nominal end point of input line

                y1 += yo #adjust with calculated offset

                #offset lines in x
                try:
                    xo = line_x_offsets[x0][e]
                except KeyError:
                    try:
                        xo = max(line_x_offsets[x0].values()) + .04
                        line_x_offsets[x0][e] = xo
                    except KeyError:
                        xo = 0
                        line_x_offsets[x0] = {e: xo}

                #print xo
                #inp_x_offsets.add(e)

                #thread trace through blocks
                xv_ = [x0]
                yv_ = [y0]

                #xvs = np.arange(x0, x1)
                #print xvs, yvs[x0:x1]
                #print 'StartTrace - y1:', y1

                for i, x_i in enumerate(range(x0 + 1, x1, 2)):
                    #print i, x_i
                    yvo = 1.0 * yv_[-1]
                    try:
                        yn = yvs_e[x_i][e]
                        recycling_y = True
                    except KeyError:
                        yn = optimize.fmin(_path_lh, yvo + .02 * np.random.randn(),
                                           (yvo, y1, np.array(yvs[x_i]), np.array(yvtol[x_i])), disp=0)[0]
                        recycling_y = False

                    xn = x_i - .4

                    #find vertical lines which might interfere
                    vl = [l[0] for l in vlines[x_i] if
                          (not l[1] == e) and (max(yn, yvo) > (min(l[0][1, :]) - .05)) and (
                              min(yn, yvo) < (max(l[0][1, :]) + .05))]
                    if len(vl) > 0:
                        xn = optimize.fmin(_path_v_lh, xn, (xn, yn, vl, np.sign(yn - yvo)), disp=0)[0]

                    if not np.isnan(yn):
                        yv_.append(yvo)
                        yv_.append(yn)
                        xv_.append(xn)
                        xv_.append(xn)
                        vlines[x_i].append((np.array([[xn, xn], [yvo, yn]]), e))
                        #print np.array([[xn, xn], [yvo, yn]])[1,:]

                        if not recycling_y:
                            yvs[x_i].append(yn)
                            yvs_e[x_i][e] = yn
                            yvtol[x_i].append(.1)
                    else:
                        if not recycling_y:
                            yvs[x_i].append(yvo)
                            yvs_e[x_i][e] = yvo
                            yvtol[x_i].append(.1)

                xn = x1 - .4
                yvo = 1.0 * yv_[-1]
                yn = y1

                #find vertical lines which might interfere
                #vl = [l for l in vlines[x1] if (yn > (min(l[1,:]) - .05)) and (yn < (max(l[1,:]) + .05))]
                #vl = [l for l in vlines[x1] if (max(yn, yvo) > (min(l[1,:]) - .05)) and (min(yn, yvo) < (max(l[1,:]) + .05))]
                vl = [l[0] for l in vlines[x1] if (not l[1] == e) and (max(yn, yvo) > (min(l[0][1, :]) - .05)) and (
                    min(yn, yvo) < (max(l[0][1, :]) + .05))]
                if len(vl) > 0:
                    xn = optimize.fmin(_path_v_lh, xn, (xn, yn, vl, np.sign(yn - yvo)), disp=0)[0]

                yv_.append(yvo)
                xv_.append(xn)
                xv_.append(xn)
                yv_.append(yn)

                vlines[x1].append((np.array([[xn, xn], [yvo, yn]]), e))

                yv_.append(y1)
                xv_.append(x1)

                connecting_lines.append((np.array(xv_), np.array(yv_), e))

    return ips, connecting_lines

def layout_vertical(dg, rdg):
    """
    Find the input lines coming in to each processing node

    Parameters
    ----------
    dg : the dependency graph

    Returns
    -------

    """
    import toposort
    from . import base
    
    ts = toposort.toposort(dg)
    
    ordered = []
    
    def _descend(node, dg, rdg):
        children = rdg.get(node, [])
        
        if isinstance(node, base.ModuleBase):
            ordered.extend(list(children))
            for child in sorted(children):
                _descend(child, dg, rdg)
        else:
            for child in sorted(children, key=lambda x: str(x)) :
                if (not child in ordered) and (np.all([d in ordered for d in dg[child]])):
                    # are all the childs dependencies already in our list of nodes?
                    ordered.append(child)
                    _descend(child, dg, rdg)
    
    for t_i in ts:
        for n in sorted(list(t_i), key=lambda x: x if isinstance(x,str) else x.__class__.__name__):
            if not n in ordered:
                ordered.append(n)
                _descend(n, dg, rdg)

    data_nodes = [n for n in ordered if not isinstance(n, base.ModuleBase)]
    data_origin = np.zeros(len(data_nodes))
    last_consumer = np.zeros(len(data_nodes))
    span_lengths = np.zeros(len(data_nodes))
    data_x = -1 * np.ones(len(data_nodes), dtype='i')

    x0s = {}

    for N, node in enumerate(data_nodes):
        consumers = rdg.get(node, [])
        data_origin[N] = ordered.index(node)
        if len(consumers) > 0:
            last_consumer[N] = max([ordered.index(c) for c in consumers])
            span_lengths[N] = last_consumer[N] - data_origin[N]
        else:
            last_consumer[N] = 0
            span_lengths[N] = 0

    span_order = np.argsort(span_lengths)

    for i in span_order:
        free_x = np.ones(len(data_nodes), dtype='bool')
    
        overlapping = (data_x >= 0) * (last_consumer > data_origin[i]) * (data_origin < last_consumer[i])
        free_x[data_x[overlapping]] = False
    
        closest_x = np.where(free_x)[0][0]
        data_x[i] = closest_x
        x0s[data_nodes[i]] = closest_x
        
    return ordered, x0s
            
def plot_vertical(recipe, fig = None):
    from matplotlib import pyplot as plt
    from .base import ModuleBase, OutputModule
    import textwrap
    #from PYME.misc.extraCMaps import labeled
    
    dg = recipe.dependancyGraph()
    rdg = recipe.reverseDependancyGraph()
    
    ordered, x0s = layout_vertical(dg, rdg)

    if fig is None:
        fig = plt.figure()
    
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1], aspect='equal')
        
    y = 0
    
    x_0 = -.05
    
    x_available = np.ones(len(ordered), dtype='bool')
    x_vals = -.1*(1+ np.arange(len(ordered)))
    
    yvs = {}
    
    output_positions = {}
    input_positions = {}

    #axisWidth = self.ax.get_window_extent().width
    #nCols = max([1] + [v[0] for v in node_positions.values()])
    pix_per_col = 200#axisWidth / float(nCols)

    fontSize = max(6, min(10, 10 * pix_per_col / 100.))

    #print pix_per_col, fontSize

    TW = textwrap.TextWrapper(width=int(1.8 * pix_per_col / fontSize), subsequent_indent='  ')
    TW2 = textwrap.TextWrapper(width=int(1.3 * pix_per_col / fontSize), subsequent_indent='  ')

    

    #Plot the connecting lines
    # for xv, yv, e in connecting_lines:
    #     #choose a colour at random for this input
    #     if not e in cols.keys():
    #         cols[e] = 0.7 * np.array(pylab.cm.hsv(pylab.rand()))
    #
    #     self.ax.plot(xv, yv, c=cols[e], lw=2)
    
    

    cols = {}
    def _col(node):
        if not hasattr(_col, '_indices'):
            _col.n_col = 0
        if not node in cols.keys():
            _col.n_col += 1
            cols[node] = 0.7 * np.array(plt.cm.hsv(np.random.rand()))
            #cols[node] = 0.7 * np.array(plt.cm.hsv((_col.n_col % len(data_nodes)) / float(len(data_nodes))))
        return cols[node]

    def _label(node, xp, yp):
        if node.startswith('out'):
            t = ax.text(xp + .1, yp + .1, node, color=_col(node), size=fontSize, weight='bold', picker=True,
                        bbox={'color': 'w', 'edgecolor': 'k'})
        else:
            t = ax.text(xp + .1, yp + .1, node, color=_col(node), size=fontSize, weight='bold', picker=True,
                        bbox={'color': 'w', 'edgecolor': 'k'})
        t._data = node
    
    for N, node in enumerate(ordered):
        yvs[node] = y
        
        if isinstance(node, ModuleBase):
            #This is a module - plot a box
            s = node.__class__.__name__
            
            if isinstance(node, OutputModule):
                fc = [.8, 1, .8]
            else:
                fc = [.8, .8, 1]
            
            #draw the box
            rect = plt.Rectangle([0, y], 1, .3, ec='k', fc=fc, picker=True)
            rect._data = node
            ax.add_patch(rect)

            #draw the title
            s = TW2.wrap(s)
            if len(s) == 1:
                ax.text(.05, y + .18, s[0], size=fontSize, weight='bold')
            else:
                ax.text(.05, y + .18 - .05 * (len(s) - 1), '\n'.join(s), size=fontSize, weight='bold')
                
            #draw the parameter text
            # s2 = []
            # for pn, p in node.get().items():
            #     if not (pn.startswith('_') or pn.startswith('input') or pn.startswith('output')):
            #         s2 += TW.wrap('%s : %s' % (pn, p))
            #
            # if len(s2) > 5:
            #     s2 = '\n'.join(s2[:4]) + '\n ...'
            # else:
            #     s2 = '\n'.join(s2)
            #
            # ax.text(.05, y - .22, s2, size=.8 * fontSize, stretch='ultra-condensed')
            
            #draw input lines
            inputs = list(node.inputs)
            ip_xs = [input_positions[ip][0] for ip in inputs]
            ip_ys = y + np.linspace(0, .3, 2 * len(inputs) + 1)[1::2]
            ip_ys = ip_ys[np.argsort(ip_xs)[::-1]]
            
            
            for ip_y, ip in zip(ip_ys, inputs):
                xi, yi = input_positions[ip]
                ax.plot([xi, xi, 0], [yi, ip_y, ip_y], '-', color=_col(ip), lw=2)
            
            
            #draw the nodes outputs
            outputs = list(node.outputs)[::-1]
            if len(outputs) > 0:
                op_y = y + np.linspace(0, .3, 2*len(outputs)+1)[1::2]
                op_x = 2 + 0.1*np.arange(len(outputs))[::-1]
                
                for yp, xp, op in zip(op_y, op_x, outputs):
                    ax.plot(1, yp, 'o', color=_col(op))
                    ax.plot([1, 2], [yp, yp], '-', color=_col(op), lw=2)
                        
                    _label(op, 1, yp - .08)
                    
                    output_positions[op] = (xp, yp)
            
            #increment our y position
            y += .35
        else:
            # we must be an input - route back to LHS
            try:
                #only map back if we are going to be used.
                if rdg.get(node, False):
                    xi, yi = output_positions[node]
                    x_0 = x_vals[x0s[node]]
                    ax.plot([2, xi, xi, x_0], [yi, yi, y, y], '-', color=_col(node), lw=2)

                    input_positions[node] = (x_0, y)
                    
                    #increment our y position
                    y += .05
                
            except KeyError:
                #dangling input
                #closest_x = np.argmax(x_vals + 100 * x_available)
                #x_available[closest_x] = False
                x_0 = x_vals[x0s[node]]
                input_positions[node] = (x_0, y)
                
                #ax.plot(1, y, 'o', color=_col(node))
                ax.plot([-1.5, x_0], [y, y], '-', color=_col(node), lw=2)
            
                _label(node, -1.4, y-.1)

                #increment our y position
                y += .2
            
            
            
            
    ax.set_ylim(y+.5, -.5)
    ax.set_xlim(-1.5, 3.0)
            

def to_svg(dg):
    import svgwrite
    node_positions, connecting_lines = layout(dg)

    ipsv = np.array(list(node_positions.values()))
    try:
        xmn, ymn = ipsv.min(0)
        xmx, ymx = ipsv.max(0)

        #self.ax.set_ylim(ymn - 1, ymx + 1)
        #self.ax.set_xlim(xmn - .5, xmx + .7)
    except ValueError:
        pass

    dwg = svgwrite.Drawing()
    vb = dwg.viewbox(xmn - .5, ymn -.5, xmx + 2, ymx + 1)
    #dwg.add(vb)
    cols = {}
    for xv, yv, e in connecting_lines:
        c = cols.get(e, None)
        if not c:
            c = 'hsl(%d' % int(255*np.random.random()) + ', 100%, 40%)'
            cols[e] = c
        dwg.add(dwg.polyline(np.vstack([xv, yv]).T, style="stroke:%s;stroke-width:.02;fill:none;" % c))

    for k, v in node_positions.items():
        if not (isinstance(k, six.string_types)):
            #node - draw a box
            #################
            s = k.__class__.__name__
            #pylab.plot(v[0], v[1], 'o', ms=5)
            #rect = pylab.Rectangle([v[0], v[1] - .25], 1, .5, ec='k', fc=[.8, .8, 1], picker=True)

            #rect._data = k
            #self.ax.add_patch(rect)
            r = dwg.add(dwg.rect((v[0], v[1] - .25), (1, .5), style="fill:#dadada;stroke:black;stroke-width:.005"))
            dwg.add(dwg.text(str(s), x=[(v[0] + .05),], y=[(v[1] - .15),], fill='black', style='font-size:0.1px;font-family:"Arial Black", Gadget, sans-serif'))

            # s = TW2.wrap(s)
            # if len(s) == 1:
            #     self.ax.text(v[0] + .05, v[1] + .18, s[0], size=fontSize, weight='bold')
            # else:
            #     self.ax.text(v[0] + .05, v[1] + .18 - .05 * (len(s) - 1), '\n'.join(s), size=fontSize, weight='bold')
            # #print repr(k)
            #
            params = k.get()
            param_names = [tn for tn in k.class_editable_traits() if not (tn.startswith('input') or tn.startswith('output'))]

            v_ = v[1] - .06
            for i, pn in enumerate(param_names):
                #s2 += TW.wrap('%s : %s' % i)
                if i > 5:
                    dwg.add(dwg.text('...', x=[(v[0] + .05), ], y=[v_, ],
                                     style='fill:#000090;font-size:0.05px;font-family:"Arial", Helvetica, sans-serif'))
                    break
                else:
                    s2 = ('%s : %s' % (pn, params[pn]))[:50]
                    dwg.add(dwg.text(str(s2), x=[(v[0] + .05), ], y=[v_, ],
                                     style='fill:#000090;font-size:0.05px;font-family:"Arial", Helvetica, sans-serif'))
                    v_ += .05




            # self.ax.text(v[0] + .05, v[1] - .22, s2, size=.8 * fontSize, stretch='ultra-condensed')
        else:
            #line - draw an output dot, and a text label
            c = cols.get(k, None)
            if not c:
                c = 'hsl(%2.3f' % np.random.random() + ', 100%, 40%)'
                cols[k] = c
            dwg.add(dwg.text(str(k), x=[(v[0] + .1), ], y=[(v[1] - .02), ],
                             style='font-size:0.1px;font-family:"Arial", Helvetica, sans-serif;fill:%s;' % c))

            pass
            # s = k
            # if not k in cols.keys():
            #     cols[k] = 0.7 * np.array(pylab.cm.hsv(pylab.rand()))
            # self.ax.plot(v[0], v[1], 'o', color=cols[k])
            # if k.startswith('out'):
            #     t = self.ax.text(v[0] + .1, v[1] + .02, s, color=cols[k], size=fontSize, weight='bold', picker=True,
            #                      bbox={'color': 'w', 'edgecolor': 'k'})
            # else:
            #     t = self.ax.text(v[0] + .1, v[1] + .02, s, color=cols[k], size=fontSize, weight='bold', picker=True)
            # t._data = k

    return dwg.tostring()
