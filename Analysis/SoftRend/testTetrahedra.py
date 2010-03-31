from pylab import *
from PYME.DSView.dsviewer_npy import View3D
from PYME.Analysis.LMVis import gen3DTriangs

x = 5e3*rand(1000)
y = 2.5e3*rand(1000)
z = 5e3*rand(1000)


im = zeros((250, 150, 25), order='F')

gen3DTriangs.renderTetrahedra(im, x, y, z, scale=[1, 1, 1], pixelsize=[10, 10, 100])
View3D(im)