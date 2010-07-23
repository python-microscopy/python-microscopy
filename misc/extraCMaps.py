import matplotlib.colors as colors
#import matplotlib as mpl
import pylab

if not 'cmapnames' in dir(pylab.cm):
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


pylab.cm.cmapnames.sort()