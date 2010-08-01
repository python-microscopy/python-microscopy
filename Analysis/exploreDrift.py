#t=colourFilter['tIndex']
#x=colourFilter['fitResults_x0']
#y=colourFilter['fitResults_y0']

from matplotlib import delaunay
from pylab import *
from numpy import *

def entlin(x,y,t,a,b):
 	x1=x+a*t
 	y1=y+b*t
 	T = delaunay.Triangulation(x1,y1)
 	return log(sqrt(diff(x1[T.edge_db])**2 + diff(y1[T.edge_db])**2)).mean()

# todo: ensure arng and brng are 1D
def entlinarr(x,y,t,arng,brng):
	ent2d = zeros((arng.size,brng.size))
	for i in range(arng.size):
		for j in range(brng.size):
			ent2d[i,j] = entlin(x,y,t,arng[i],brng[j])
	return ent2d

def entlin2d(x,y,t,alim,blim,size):
	arng = linspace(alim[0],alim[1],size[0])
	brng = linspace(blim[0],blim[1],size[1])
	return entlinarr(x,y,t,arng,brng)

def entlin2dr(x,y,t,alim,blim,size):
	arng = linspace(alim[0],alim[1],size[0])
	brng = linspace(blim[0],blim[1],size[1])
	return entlinarr(x+rand(x.size),y+rand(y.size),t,arng,brng)




# from PYME.Analysis.exploreDrift import *
# e2d = entlin2dr(filter['x'],filter['y'],filter['t'],[-0.05,0.05],[-0.05,0.05],[20,20])
# imshow(e2d,interpolation='nearest')
