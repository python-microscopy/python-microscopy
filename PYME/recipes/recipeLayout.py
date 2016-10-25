import numpy as np
from PYME.misc import depGraph
from scipy import optimize

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
        if not (isinstance(k, str) or isinstance(k, unicode)):
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


def to_svg(dg):
    import svgwrite
    node_positions, connecting_lines = layout(dg)

    ipsv = np.array(node_positions.values())
    try:
        xmn, ymn = ipsv.min(0)
        xmx, ymx = ipsv.max(0)

        #self.ax.set_ylim(ymn - 1, ymx + 1)
        #self.ax.set_xlim(xmn - .5, xmx + .7)
    except ValueError:
        pass

    dwg = svgwrite.Drawing()
    vb = dwg.viewbox(xmn - .5, ymn -1, xmx + 1, ymx + 2)
    #dwg.add(vb)
    cols = {}
    for xv, yv, e in connecting_lines:
        c = cols.get(e, None)
        if not c:
            c = 'hsl(%d' % int(255*np.random.random()) + ', 100%, 40%)'
            cols[e] = c
        dwg.add(dwg.polyline(np.vstack([xv, yv]).T, style="stroke:%s;stroke-width:.02;fill:none;" % c))

    for k, v in node_positions.items():
        if not (isinstance(k, str) or isinstance(k, unicode)):
            #node - draw a box
            #################
            s = k.__class__.__name__
            #pylab.plot(v[0], v[1], 'o', ms=5)
            #rect = pylab.Rectangle([v[0], v[1] - .25], 1, .5, ec='k', fc=[.8, .8, 1], picker=True)

            #rect._data = k
            #self.ax.add_patch(rect)
            r = dwg.add(dwg.rect((v[0], v[1] - .25), (1, .5), style="fill:#9a9a9a;"))
            dwg.add(dwg.text(str(s), x=[(v[0] + .05),], y=[(v[1] + .18),], fill='black', style='font-size:0.1px;font-family:"Arial Black", Gadget, sans-serif'))

            # s = TW2.wrap(s)
            # if len(s) == 1:
            #     self.ax.text(v[0] + .05, v[1] + .18, s[0], size=fontSize, weight='bold')
            # else:
            #     self.ax.text(v[0] + .05, v[1] + .18 - .05 * (len(s) - 1), '\n'.join(s), size=fontSize, weight='bold')
            # #print repr(k)
            #
            # params = k.get().items()
            # s2 = []
            # for i in params:
            #     s2 += TW.wrap('%s : %s' % i)
            #
            # if len(s2) > 5:
            #     s2 = '\n'.join(s2[:4]) + '\n ...'
            # else:
            #     s2 = '\n'.join(s2)
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