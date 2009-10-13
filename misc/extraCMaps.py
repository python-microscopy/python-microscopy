import matplotlib.colors as colors
#import matplotlib as mpl
import pylab

_r = {'red':((0.,0.,0.), (1.,1.,1.)), 'green':((0.,0,0), (1.,0.,0.)), 'blue':((0.,0.,0.), (1.,0.,0.))}
_g = {'green':((0.,0.,0.), (1.,1.,1.)), 'red':((0.,0,0), (1.,0.,0.)), 'blue':((0.,0.,0.), (1.,0.,0.))}
_b = {'blue':((0.,0.,0.), (1.,1.,1.)), 'green':((0.,0,0), (1.,0.,0.)), 'red':((0.,0.,0.), (1.,0.,0.))}

ndat = {'r':_r, 'g':_g, 'b':_b}

ncmapnames = ndat.keys()
pylab.cm.cmapnames += ncmapnames
for cmapname in ncmapnames:
    pylab.cm.__dict__[cmapname] = colors.LinearSegmentedColormap(cmapname, ndat[cmapname], pylab.cm.LUTSIZE)
    cmapname_r = cmapname+'_r'
    cmapdat_r = pylab.cm.revcmap(ndat[cmapname])
    ndat[cmapname_r] = cmapdat_r
    pylab.cm.__dict__[cmapname_r] = colors.LinearSegmentedColormap(cmapname_r, cmapdat_r, pylab.cm.LUTSIZE)