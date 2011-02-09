#!/usr/bin/python
##################
# VolProj.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import pylab
from enthought.mayavi import mlab
import os
import tempfile
from PYME.FileUtils.readTiff import read3DTiff

def GetImage(figure):
    tfname = os.path.join(tempfile.gettempdir(), 'mvi_out.tif')
    f.scene.save_tiff(tfname)

    im = read3DTiff(tfname).squeeze()

    os.remove(tfname)

    return im


class Isosurf:
    def __init__(self,channels, thresholds, pixelsize = (1., 1., 1.)):
        self.f = mlab.figure()

        self.isos = []
        self.projs = []

        for im, th, i in zip(channels, thresholds, range(len(channels))):
            ps = []

            pr = im.mean(2)
            ps.append(self.drawProjection((255*pylab.minimum(pr/(1.5*th), 1)).astype('uint8'), 'z'))
            pr = im.mean(0)
            ps.append(self.drawProjection((255*pylab.minimum(pr/(1.5*th), 1)).astype('uint8'), 'x'))
            pr = im.mean(1)
            ps.append(self.drawProjection((255*pylab.minimum(pr/(1.5*th), 1)).astype('uint8'), 'y'))

            self.projs.append(ps)

            c = mlab.contour3d(im, contours=[th], color = pylab.cm.gist_rainbow(float(i)/len(channels))[:3])
            c.mlab_source.dataset.spacing = pixelsize
            self.isos.append(c)


    def drawProjection(im, axis):
        gp = mlab.pipeline.grid_plane(c)
        gp.grid_plane.axis = axis
        gp.actor.enable_texture = True
        gp.actor.texture_source_object = mlab.pipeline.scalar_field(im)
        gp.actor.tcoord_generator_mode = 'plane'
        gp.actor.property.representation = 'surface'

        return gp

    def ShowProj(self, chan, show=True):
        for p in self.projs[chan]:
            p.visible = show

    def ShowIso(self, show=True):
        for c in self.isos:
            c.visible = True
