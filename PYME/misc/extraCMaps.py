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
from matplotlib import cm
#import matplotlib as mpl
# import pylab
import numpy as np

def regCmap(cmap):
    cm.__dict__[cmap.name] = cmap
    cm.cmapnames.append(cmap.name)
    
#define a decorator
def register_cmap(name):
    def _reg_cmap(func):
        func.name = name
        regCmap(func)
        return func
    
    return _reg_cmap

if not 'cmapnames' in dir(cm):
    if 'cmap_d' in dir(cm):
        cm.cmapnames = list(cm.cmap_d.keys())
    else:
        cm.cmapnames = cm._cmapnames

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

ncmapnames = list(ndat.keys())
cm.cmapnames += ncmapnames
for cmapname in ncmapnames:
    ncm = colors.LinearSegmentedColormap(cmapname, ndat[cmapname], cm.LUTSIZE)
    cm.__dict__[cmapname] = ncm
    cm.__dict__[cmapname+'_r']=ncm.reversed()

#solid colour colormaps for VisGUI multichannel and isosurface display
@register_cmap('R')
def Red(data):
    z = np.ones_like(data)
    return np.stack([z, 0*z, 0*z, z], -1)

@register_cmap('G')
def Green(data):
    z = np.ones_like(data)
    return np.stack([0*z, z, 0*z, z], -1)

@register_cmap('B')
def Blue(data):
    z = np.ones_like(data)
    return np.stack([0*z, 0*z, z, z], -1)

@register_cmap('C')
def Cyan(data):
    z = np.ones_like(data)
    return np.stack([0*z, z, z, z], -1)

@register_cmap('M')
def Magenta(data):
    z = np.ones_like(data)
    return np.stack([z, 0*z, z, z], -1)

@register_cmap('Y')
def Yellow(data):
    z = np.ones_like(data)
    return np.stack([z, z, 0*z, z], -1)

@register_cmap('labeled')
def labeled(data):
    return (data > 0).reshape(list(data.shape) +  [1])*cm.gist_rainbow(data % 1)

@register_cmap('flow_gray')
def flow_gray(data):
    v = ((data > 0)*(data < 1)).reshape(list(data.shape) +  [1])*cm.gray(data)
    v += (data == 0).reshape(list(data.shape) + [1]) * np.array([0, 1., 0, 0]).reshape(list(np.ones(data.ndim) + [3]))
    v += (data == 1).reshape(list(data.shape) + [1]) * np.array([1., 0, 1., 0]).reshape(list(np.ones(data.ndim) + [3]))
    return v


def grey_overflow(underflowcol = 'magenta', overflowcol = 'lime', percentage=5, greystart=0.1):
    if percentage < 1:
        percentage = 1
    if percentage > 15:
        percentage = 15

    ucolrgb = colors.hex2color(colors.cnames[underflowcol])
    ocolrgb = colors.hex2color(colors.cnames[overflowcol])
    p = 0.01 * percentage

    def r(rgb):
        return rgb[0]

    def g(rgb):
        return rgb[1]

    def b(rgb):
        return rgb[2]
    
    grey_data = {'red':   [(0, r(ucolrgb), r(ucolrgb)),
                          (p, r(ucolrgb), greystart),
                          (1.0-p, 1.0, r(ocolrgb)),
                          (1.0, r(ocolrgb), r(ocolrgb))],
                'green': [(0, g(ucolrgb), g(ucolrgb)),
                          (p, g(ucolrgb), greystart),
                          (1.0-p, 1.0, g(ocolrgb)),
                          (1.0, g(ocolrgb), g(ocolrgb))],
                'blue':  [(0, b(ucolrgb), b(ucolrgb)),
                          (p, b(ucolrgb), greystart),
                          (1.0-p, 1.0, b(ocolrgb)),
                          (1.0, b(ocolrgb), b(ocolrgb))]}

    cm_grey2 = colors.LinearSegmentedColormap('grey_overflow', grey_data)
    return cm_grey2

regCmap(grey_overflow(percentage=2.5,greystart=0.125))

try:
    import PYMEcs.experimental.ExtraCmaps as ec
    regCmap(ec.hot_overflow(overflowcol='cyan',percentage=2.5))
except:
    pass

cm.cmapnames.sort()
