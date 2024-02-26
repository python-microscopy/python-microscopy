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
import numpy as np
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

        self.AddMenuItem('File', 'Save mesh', lambda e: self.save_surface())
        
        self.Layout()

    def _layout(self, *args, **kwargs):
        self.Layout()

    def save_surface(self):
        #from PYME.experimental import _triangle_mesh as triangle_mesh
        
        surf_keys = [l.dsname for l in self.canvas.layers] # if isinstance(l, triangle_mesh.TriangleMesh)]
        
        if len(surf_keys) == 0:
            raise RuntimeError('No surfaces present')
        
        if len(surf_keys) == 1:
            key = 0
        else:
            dlg = wx.SingleChoiceDialog(self, "Which surface do you want to save?", "Choose a surface to save", surf_keys)
            
            if not dlg.ShowModal():
                dlg.Destroy()
                return
            else:
                key = surf_keys.index(dlg.GetStringSelection())
                dlg.Destroy()

        filename = wx.FileSelector('Save surface as...',
                                default_extension='stl',
                                wildcard='STL mesh (*.stl)|*.stl|PLY mesh (*.ply)|*.ply',
                                flags=wx.FD_SAVE)

        if not filename == '':
            ext = filename.split('.')[-1]
            if ext == 'stl':
                self.canvas.layers[key].datasource.to_stl(filename)
            elif ext == 'ply':
                import numpy as np
                colors = None
                # If we have, save the PLY with its colors
                layer = self.canvas.layers[key]
                # Construct a re-indexing for non-negative vertices
                live_vertices = np.flatnonzero(layer.datasource._vertices['halfedge'] != -1)
                new_vertex_indices = np.arange(live_vertices.shape[0])
                vertex_lookup = np.zeros(layer.datasource._vertices.shape[0])
                
                vertex_lookup[live_vertices] = new_vertex_indices

                # Grab the faces and vertices we want
                faces = vertex_lookup[layer.datasource.faces]

                colors = np.zeros((live_vertices.size, 3), dtype=np.ubyte)
                colors[faces.ravel().astype(int)] = np.floor(layer._colors[:,:3]*255).astype(np.ubyte)
                    
                layer.datasource.to_ply(filename, colors)
            else:
                raise ValueError('Invalid file extension .' + str(ext))
        
        
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
        dsviewer.AddMenuItem('&3D', '3D Isosurface (coloured)', self.isosurf_coloured)
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
        from PYME.misc.colormaps import cm

        glcanvas = new_mesh_viewer()#glrender.showGLFrame()
        glcanvas.layer_data={}

        for i in range(self.image.data.shape[3]):
            isolevel = self.do.Offs[i] + .5/self.do.Gains[i]
            T = isosurface.isosurface(self.image.data[:,:,:,i].astype('f'), isolevel=isolevel, voxel_size=self.image.voxelsize, origin=self.image.origin)
            glcanvas.layer_data[self.image.names[i]] = T
            layer = TriangleRenderLayer(glcanvas.layer_data, dsname=self.image.names[i], method='shaded', context=glcanvas.gl_context, window = glcanvas,
                                        cmap=cm.solid_cmaps[i % len(cm.solid_cmaps)],
                                        #normal_mode='Per face', #use face normals rather than vertex normals, as there is currently a bug in computation of vertex normals
                                        )
            glcanvas.add_layer(layer)
            
            layer.engine._outlines=False
            layer.show_lut=False
            
        
        self.canvases.append(glcanvas)
        
        glcanvas.displayMode = '3D'
        glcanvas.fit_bbox()
        glcanvas.Refresh()

    def isosurf_coloured(self, event=None):
        from PYME.experimental import isosurface
        from PYME.LMVis import gl_render3D_shaders as glrender
        from PYME.LMVis.layers.mesh import TriangleRenderLayer
        from PYME.misc.colormaps import cm

        glcanvas = new_mesh_viewer()#glrender.showGLFrame()
        glcanvas.layer_data={}

        # assume first channel intensity, second colour
        # TODO - make this more flexible
        assert(self.image.data_xyztc.shape[4] ==2)
        
        i = 0 # use first channel for threshold
        isolevel = self.do.Offs[i] + .5/self.do.Gains[i]
        T = isosurface.isosurface(self.image.data_xyztc[:,:,:, 0, i].astype('f'), isolevel=isolevel, voxel_size=self.image.voxelsize, origin=self.image.origin)
        
        # get pixel locations of the vertices
        xi, yi, zi = (T.vertices/np.array(self.image.voxelsize)).astype('i').T
        
        d = self.image.data_xyztc[:, :, :, 0, 1].squeeze()
        xi = np.clip(xi, 0, d.shape[0]-1)
        yi = np.clip(yi, 0, d.shape[1]-1)
        zi = np.clip(zi, 0, d.shape[2]-1)
        c = d[xi, yi, zi].squeeze()
        T.extra_vertex_data['cval'] = c
        
        glcanvas.layer_data[self.image.names[i]] = T
        layer = TriangleRenderLayer(glcanvas.layer_data, dsname=self.image.names[i], method='shaded', context=glcanvas.gl_context, window = glcanvas,
                                    cmap='jet', vertexColor='cval',
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



