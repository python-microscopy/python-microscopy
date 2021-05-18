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
# import pylab
import matplotlib.cm

from PYME.ui.AUIFrame import AUIFrame
import wx.lib.agw.aui as aui
from PYME.LMVis import view_clipping_pane

class MeshViewFrame(AUIFrame):
    def __init__(self, *args, **kwargs):
        from PYME.LMVis import gl_render3D_shaders as glrender
        from PYME.LMVis import layer_panel
        AUIFrame.__init__(self, *args, **kwargs)
        
        self.canvas = glrender.LMGLShaderCanvas(self)
        self.AddPage(page=self.canvas, caption='View')

        
        self.panesToMinimise = []
        
        self.layerpanel = layer_panel.LayerPane(self, self.canvas, caption=None, add_button=False)
        self.layerpanel.SetSize(self.layerpanel.GetBestSize())
        pinfo = aui.AuiPaneInfo().Name("layerPanel").Right().Caption('Layers').CloseButton(
            False).MinimizeButton(True).MinimizeMode(
            aui.AUI_MINIMIZE_CAPT_SMART | aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        self._mgr.AddPane(self.layerpanel, pinfo)

        self.panesToMinimise.append(pinfo)
        
        self._clip_pane = view_clipping_pane.ViewClippingPanel(self, self.canvas)
        pinfo = aui.AuiPaneInfo().Name("ClippingPanel").Right().Caption('Clipping').CloseButton(
            False).MinimizeButton(True).MinimizeMode(
            aui.AUI_MINIMIZE_CAPT_SMART | aui.AUI_MINIMIZE_POS_RIGHT)
        self._mgr.AddPane(self._clip_pane, pinfo)

        self.panesToMinimise.append(pinfo)

        # self._mgr.AddPane(self.optionspanel.CreateToolBar(self),
        #                   aui.AuiPaneInfo().Name("ViewTools").Caption("View Tools").CloseButton(False).
        #                   ToolbarPane().Right().GripperTop())
        
        for pn in self.panesToMinimise:
            self._mgr.MinimizePane(pn)
        
        self.Layout()
        
        
def new_mesh_viewer(parent=None,*args, **kwargs):
    kwargs['size'] = kwargs.get('size', (800,800))
    f = MeshViewFrame(parent, *args, **kwargs)
    f.Show()

    f.canvas.SetCurrent(f.canvas.gl_context)
    f.canvas.initialize()
    return f.canvas
        
from ._base import Plugin
class Visualiser(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        self.tq = None
        
        self.canvases = []
        
        dsviewer.AddMenuItem('&3D', '3D Isosurface (mayavi)', self.On3DIsosurf)
        dsviewer.AddMenuItem('&3D', '3D Isosurface (builtin)', self.isosurf_builtin)
        dsviewer.AddMenuItem('&3D', '3D Volume', self.On3DVolume)
        dsviewer.AddMenuItem('&3D', 'Save Isosurface as STL', self.save_stl)
        dsviewer.AddMenuItem('&3D', 'Save Isosurface(s) as u3d', self.save_u3d)
        dsviewer.AddMenuItem('&3D', 'Save Isosurface(s) as x3d', self.save_x3d)
        


    def On3DIsosurf(self, event):
        try:
            from enthought.mayavi import mlab
        except ImportError:
            from mayavi import mlab

        self.dsviewer.f3d = mlab.figure()
        self.dsviewer.f3d.scene.stereo = True

        for i in range(self.image.data.shape[3]):
            c = mlab.contour3d(self.image.data[:,:,:,i].squeeze().astype('f'), contours=[self.do.Offs[i] + .5/self.do.Gains[i]], color = matplotlib.cm.gist_rainbow(float(i)/self.image.data.shape[3])[:3])
            self.lastSurf = c
            c.mlab_source.dataset.spacing = self.image.voxelsize
            
    def isosurf_builtin(self, event=None):
        from PYME.experimental import isosurface
        from PYME.LMVis import gl_render3D_shaders as glrender
        from PYME.LMVis.layers.mesh import TriangleRenderLayer

        glcanvas = new_mesh_viewer()#glrender.showGLFrame()
        glcanvas.layer_data={}

        for i in range(self.image.data.shape[3]):
            isolevel = self.do.Offs[i] + .5/self.do.Gains[i]
            T = isosurface.isosurface(self.image.data[:,:,:,i].astype('f'), isolevel=isolevel, voxel_size=self.image.voxelsize, origin=self.image.origin)
            glcanvas.layer_data[self.image.names[i]] = T
            layer = TriangleRenderLayer(glcanvas.layer_data, dsname=self.image.names[i], method='shaded', context=glcanvas.gl_context,
                                        cmap=['C', 'M', 'Y', 'R', 'G', 'B'][i % 6],
                                        #normal_mode='Per face', #use face normals rather than vertex normals, as there is currently a bug in computation of vertex normals
                                        )
            glcanvas.add_layer(layer)
            
            layer.engine._outlines=False
            layer.show_lut=False
            
        
        self.canvases.append(glcanvas)
        
        glcanvas.displayMode = '3D'
        glcanvas.fit_bbox()
        glcanvas.Refresh()
            
        



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
        
        fdialog = wx.FileDialog(None, 'Save 3D scene as ...', wildcard='*.stl', style=wx.FD_SAVE)#|wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
        
        if (succ == wx.ID_OK):
            fname = fdialog.GetPath()
            tvtk.STLWriter(input=self.lastSurf.actor.mapper.input, file_name=fname).write()

        fdialog.Destroy()

    def save_u3d(self, event=None):
        """Save last renderd scene as u3d."""
        from tvtk.api import tvtk
        try:
            import vtku3dexporter
        except ImportError:
            wx.MessageBox('u3d export needs the vtku3dexporter module, which is not installed by default with PYME\n A conda-installable package is available for OSX.')
    
        fdialog = wx.FileDialog(None, 'Save 3D scene as ...', wildcard='*.u3d', style=wx.FD_SAVE)# | wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
    
        if (succ == wx.ID_OK):
            fname = fdialog.GetPath()
            
            #tvtk.STLWriter(input=self.lastSurf.actor.mapper.input, file_name=fname).write()
            render_window = tvtk.to_vtk(self.dsviewer.f3d.scene.render_window)

            u3d_exporter = vtku3dexporter.vtkU3DExporter()
            u3d_exporter.SetFileName(fname)
            u3d_exporter.SetInput(render_window)
            u3d_exporter.Write()
    
        fdialog.Destroy()

    def save_x3d(self, event=None):
        """Save last renderd scene as u3d."""
        from tvtk.api import tvtk
           
        fdialog = wx.FileDialog(None, 'Save 3D scene as ...', wildcard='*.x3d', style=wx.FD_SAVE)# | wx.HIDE_READONLY)
        succ = fdialog.ShowModal()
    
        if (succ == wx.ID_OK):
            fname = fdialog.GetPath()
        
            #tvtk.STLWriter(input=self.lastSurf.actor.mapper.input, file_name=fname).write()
            render_window = self.dsviewer.f3d.scene.render_window
        
            x3d_exporter = tvtk.X3DExporter(file_name=fname, input=render_window, binary=0)
            x3d_exporter.write()
    
        fdialog.Destroy()
            


def Plug(dsviewer):
    return Visualiser(dsviewer)



