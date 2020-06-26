#!/usr/bin/python
##################
# VolProj.py
#
# Copyright David Baddeley, 2011
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
##################
# import pylab
import numpy as np
import matplotlib.cm
try:
    from enthought.mayavi import mlab
except ImportError:
    from mayavi import mlab
import os
import tempfile
#from PYME.IO.FileUtils.readTiff import read3DTiff

try:
    import Image
except ImportError:
    from PIL import Image

def GrabImage(figure):
    tfname = os.path.join(tempfile.gettempdir(), 'mvi_out.tif')
    figure.scene.save_tiff(tfname)

    im = Image.open(tfname)

    ima = np.array(im.getdata()).reshape((im.size[1], im.size[0], 3))

    #im = read3DTiff(tfname).squeeze()

    os.remove(tfname)

    return ima

#COLOURS= [(1., 0., 0.), (0., 1., 0.), ]


class Isosurf:
    def __init__(self,channels, thresholds, pixelsize = (1., 1., 1.)):
        self.f = mlab.figure()

        self.isos = []
        self.projs = []

        for im, th, i in zip(channels, thresholds, range(len(channels))):
            c = mlab.contour3d(im, contours=[th], color = matplotlib.cm.gist_rainbow(float(i)/len(channels))[:3])
            c.mlab_source.dataset.spacing = pixelsize
            self.isos.append(c)

            ps = []
            thf = th*1.5

            pr = im.mean(2)
            #f = im.max()/pr.max()
            #pr *= im.max()/pr.max()
            ps.append(self.drawProjection((255*np.minimum(pr/(1.*thf), 1)).astype('uint8'), 'z', c))
            pr = im.mean(0)
            #pr *= im.max()/pr.max()
            ps.append(self.drawProjection((255*np.minimum(pr/(.6*thf), 1)).astype('uint8'), 'x', c))
            pr = im.mean(1)
            #pr *= im.max()/pr.max()
            ps.append(self.drawProjection((255*np.minimum(pr/(.8*thf), 1)).astype('uint8'), 'y', c))

            self.projs.append(ps)




    def drawProjection(self, im, axis, c):
        gp = mlab.pipeline.grid_plane(c)
        gp.actor.property.ambient=1.0
        gp.actor.property.diffuse=0.0
        #gp.visible = False
        gp.grid_plane.axis = axis
        gp.actor.enable_texture = True
        gp.actor.texture_source_object = mlab.pipeline.scalar_field(im)
        gp.actor.tcoord_generator_mode = 'plane'
        gp.actor.property.representation = 'surface'

        return gp

    def ShowProj(self, chan, show=True):
        if chan < len(self.projs):
            for p in self.projs[chan]:
                p.visible = show

    def ShowIso(self, show=True):
        for c in self.isos:
            c.visible = show

    def Grab(self):
        return GrabImage(self.f)

    def GrabComp(self):
        for i in range(len(self.projs)):
            self.ShowProj(i, False)

        surfs = self.Grab()
        mask = (surfs == 127).prod(2)

        ps = []
        psmask = mask.copy()

        self.ShowIso(False)

        for i in range(min(len(self.projs), 3)):
            self.ShowProj(i, True)
            p_i = self.Grab()
            psmask *= (p_i ==127).prod(2)
            ps.append(p_i[:,:,i])
            self.ShowProj(i, False)

        self.ShowIso(True)
        for i in range(len(self.projs)):
            self.ShowProj(i, True)

        im = 0*surfs

        for i in range(3):
            if i < len(ps):
                im_i = ps[i]*mask
            else:
                im_i = 127*psmask
            im[:,:,i] = im_i + surfs[:,:,i]*(1-mask)


        return im
