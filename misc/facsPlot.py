from pylab import *
from scipy import ndimage

def facsPlotScatter(x, y, nbins=None):
    if nbins == None:
        nbins = 0.25*sqrt(len(x))
    n, xedge, yedge = histogram2d(x, y, bins = [nbins,nbins], range=[(min(x), max(x)), (min(y), max(y))])

    dx = diff(xedge[:2])
    dy = diff(yedge[:2])

    c = ndimage.map_coordinates(n, [(x - xedge[0])/dx, (y - yedge[0])/dy])

    scatter(x, y, c=c, s=1, edgecolors='none')