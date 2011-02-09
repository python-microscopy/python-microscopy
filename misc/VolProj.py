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





class Isosurf:
    def __init__(self,channels, thresholds, pixelsize = (1., 1., 1.)):
        self.f = mlab.figure()

        self.isos = []


        for im, th, i in zip(channels, thresholds, range(len(channels))):
            projs

            pr = im.mean(2)
            zproj.append((255*pylab.minimum(pr/(1.5*th), 1)).astype('uint8'))
            pr = im.mean(0)
            xproj.append((255*pylab.minimum(pr/(1.5*th), 1)).astype('uint8'))
            pr = im.mean(1)
            yproj.append((255*pylab.minimum(pr/(1.5*th), 1)).astype('uint8'))

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
