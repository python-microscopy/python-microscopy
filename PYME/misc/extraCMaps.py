#!/usr/bin/python

###############
# extraCMaps.py
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
import matplotlib.colors as colors
#import matplotlib as mpl
import pylab

def regCmap(cmap):
    pylab.cm.__dict__[cmap.name] = cmap
    pylab.cm.cmapnames.append(cmap.name)

if not 'cmapnames' in dir(pylab.cm):
    if 'cmap_d' in dir(pylab.cm):
        pylab.cm.cmapnames = pylab.cm.cmap_d.keys()
    else:
        pylab.cm.cmapnames = pylab.cm._cmapnames

_r = {'red':((0.,0.,0.), (1.,1.,1.)), 'green':((0.,0,0), (1.,0.,0.)), 'blue':((0.,0.,0.), (1.,0.,0.))}
_g = {'green':((0.,0.,0.), (1.,1.,1.)), 'red':((0.,0,0), (1.,0.,0.)), 'blue':((0.,0.,0.), (1.,0.,0.))}
_b = {'blue':((0.,0.,0.), (1.,1.,1.)), 'green':((0.,0,0), (1.,0.,0.)), 'red':((0.,0.,0.), (1.,0.,0.))}

_c = {'blue':((0.,0.,0.), (1.,1.,1.)), 'green':((0.,0,0), (1.,1.,1.)), 'red':((0.,0.,0.), (1.,0.,0.))}
_m = {'blue':((0.,0.,0.), (1.,1.,1.)), 'red':((0.,0,0), (1.,1.,1.)), 'green':((0.,0.,0.), (1.,0.,0.))}
_y = {'red':((0.,0.,0.), (1.,1.,1.)), 'green':((0.,0,0), (1.,1.,1.)), 'blue':((0.,0.,0.), (1.,0.,0.))}

_hsv_part = {'red':   ((0., 1., 1.),(0.25, 1.000000, 1.000000),
                       (0.5, 0, 0),
                       (1, 0.000000, 0.000000)),
             'green': ((0., 0., 0.),(0.25, 1, 1),
                       (.75, 1.000000, 1.000000),
                       (1.0, 0.2, 0.2)),
             'blue':  ((0., 0., 0.),(0.5, 0.000000, 0.000000),
                       (0.75, 1, 1),
                       (1.0, 1, 1))}

ndat = {'r':_r, 'g':_g, 'b':_b, 'c':_c, 'm':_m, 'y':_y, 'hsp': _hsv_part}

ncmapnames = ndat.keys()
pylab.cm.cmapnames += ncmapnames
for cmapname in ncmapnames:
    pylab.cm.__dict__[cmapname] = colors.LinearSegmentedColormap(cmapname, ndat[cmapname], pylab.cm.LUTSIZE)
    cmapname_r = cmapname+'_r'
    cmapdat_r = pylab.cm.revcmap(ndat[cmapname])
    ndat[cmapname_r] = cmapdat_r
    pylab.cm.__dict__[cmapname_r] = colors.LinearSegmentedColormap(cmapname_r, cmapdat_r, pylab.cm.LUTSIZE)

def labeled(data):
    return (data > 0).reshape(list(data.shape) +  [1])*pylab.cm.gist_rainbow(data % 1)

labeled.name = 'labeled'

regCmap(labeled)

def flow_gray(data):
    v = ((data > 0)*(data < 1)).reshape(list(data.shape) +  [1])*pylab.cm.gray(data)
    v += (data == 0).reshape(list(data.shape) + [1]) * pylab.array([0, 1., 0, 0]).reshape(list(pylab.ones(data.ndim) + [3]))
    v += (data == 1).reshape(list(data.shape) + [1]) * pylab.array([1., 0, 1., 0]).reshape(list(pylab.ones(data.ndim) + [3]))
    return v

flow_gray.name = 'flow_gray'

regCmap(flow_gray)


pylab.cm.cmapnames.sort()