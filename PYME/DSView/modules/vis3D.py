#!/usr/bin/python
##################
# vis3D.py
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
import numpy
import wx
import pylab

class visualiser:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        self.do = dsviewer.do

        self.image = dsviewer.image
        self.tq = None
        
        dsviewer.AddMenuItem('&3D', '3D Isosurface', self.On3DIsosurf)
        dsviewer.AddMenuItem('&3D', '3D Volume', self.On3DVolume)
        dsviewer.AddMenuItem('&3D', 'Save Isosurface as STL', self.save_stl)
        


    def On3DIsosurf(self, event):
        try:
            from enthought.mayavi import mlab
        except ImportError:
            from mayavi import mlab

        self.dsviewer.f3d = mlab.figure()
        self.dsviewer.f3d.scene.stereo = True

        for i in range(self.image.data.shape[3]):
            c = mlab.contour3d(self.image.data[:,:,:,i].astype('f'), contours=[self.do.Offs[i] + .5/self.do.Gains[i]], color = pylab.cm.gist_rainbow(float(i)/self.image.data.shape[3])[:3])
            self.lastSurf = c
            c.mlab_source.dataset.spacing = (self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z'))
            

    def On3DVolume(self, event):
        try:
            from enthought.mayavi import mlab
        except ImportError:
            from mayavi import mlab

        self.dsviewer.f3d = mlab.figure()
        self.dsviewer.f3d.scene.stereo = True

        for i in range(self.image.data.shape[3]):
            #c = mlab.contour3d(im.img, contours=[pylab.mean(ivp.clim)], color = pylab.cm.gist_rainbow(float(i)/len(self.images))[:3])
            v = mlab.pipeline.volume(mlab.pipeline.scalar_field(numpy.minimum(255*(self.image.data[:,:,:,i] -self.do.Offs[i])*self.do.Gains[i], 254).astype('uint8')))
            #v.volume.scale = (self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z'))
            
    def save_stl(self, event=None):
        """Save last renderd scene as STL."""
        from tvtk.api import tvtk
        
        fdialog = wx.FileDialog(None, 'Save 3D scene as ...', wildcard='*.stl', style=wx.SAVE|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        
        if (succ == wx.ID_OK):
            fname = fdialog.GetPath().encode()
            tvtk.STLWriter(input=self.lastSurf.actor.mapper.input, file_name=fname).write()

        fdialog.Destroy()
            


def Plug(dsviewer):
    dsviewer.vis3D = visualiser(dsviewer)



