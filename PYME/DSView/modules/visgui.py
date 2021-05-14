#!/usr/bin/python
##################
# visgui.py
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

from PYME.LMVis.imageView2 import ImageViewPanel, ColourImageViewPanel
# import pylab
import matplotlib.cm
import numpy
import wx
import os

from PYME.DSView.arrayViewPanel import ArrayViewPanel
from PYME.LMVis.gl_render3D_shaders import LMGLShaderCanvas

from six.moves import xrange

class visGuiExtras:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        
        dsviewer.AddMenuItem("View", "E&xport Current View", self.OnExport)
        dsviewer.AddMenuItem("View", "Set as visualisation &background", self.OnViewBackground)

    def OnViewBackground(self, event):
        from PYME.LMVis.layers import image_layer
        img = self.dsviewer.image
        glCanvas = self.dsviewer.glCanvas
        
        glCanvas.SetCurrent(glCanvas.gl_context) #make sure that the context we want to add the shaders to is current

        for name, i in zip(img.names, xrange(img.data.shape[3])):
            l_i = image_layer.ImageRenderLayer({'im': img}, dsname='im',
                                               display_opts=self.dsviewer.do, #slave the display scaling to the image viewer scaling
                                               channel=i,
                                               context=glCanvas.gl_context)
    
            glCanvas.layers.insert(0, l_i) #prepend layers so they are drawn before points
            
        # FIXME - this is gross - just add to glCanvas and have it issue a signal which can be caught higher up
        glCanvas.GetParent().GetParent().GetParent().layer_added.send(None)

        glCanvas.Refresh()
        



    def OnExport(self, event):
        #ivp = self.notebook.GetPage(self.notebook.GetSelection())
        ivp = self.dsviewer.GetSelectedPage()
        fname = wx.FileSelector('Save Current View', default_extension='.tif', wildcard="Supported Image Files (*.tif, *.bmp, *.gif, *.jpg, *.png)|*.tif;*.bmp;*.gif;*.jpg;*.png", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            ext = os.path.splitext(fname)[-1]
            if ext == '.tif':
                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_TIF)
            elif ext == '.png':
                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_PNG)
            elif ext == '.jpg':
                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_JPG)
            elif ext == '.gif':
                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_GIF)
            elif ext == '.bmp':
                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_BMP)
                
                
class GLImageView(LMGLShaderCanvas):
    def __init__(self, parent, image, glCanvas, display_opts, show_lut=False, **kwargs):
        LMGLShaderCanvas.__init__(self, parent=parent, show_lut=show_lut, view=glCanvas.view, **kwargs)
        
        #cmaps = ['r', 'g', 'b']
        
        self._do = display_opts
        self._do.WantChangeNotification.append(self._sync_display_opts)
        
        self.wantViewChangeNotification.add(glCanvas)
        
        self._image = image
        
        wx.CallLater(500, self._add_layers)
        
        
    def _add_layers(self):
        from PYME.LMVis.layers import image_layer
        self.SetCurrent(self.gl_context)
        for name, i in zip(self._image.names, xrange(self._image.data_xyztc.shape[4])):
            l_i = image_layer.ImageRenderLayer({'im': self._image}, dsname='im', display_opts=self._do, channel=i, context=self.gl_context)
        
            self.layers.append(l_i)
    
        self._sync_display_opts()
        
    def _sync_display_opts(self):
        for l in self.layers:
            l.sync_to_display_opts()
            
        self.Refresh()
        
def Plug(dsviewer):
    dsviewer.vgextras = visGuiExtras(dsviewer)

    cmaps = [matplotlib.cm.r, matplotlib.cm.g, matplotlib.cm.b]

    if not 'ivps' in dir(dsviewer):
        dsviewer.ivps = []

    for name, i in zip(dsviewer.image.names, xrange(dsviewer.image.data_xyztc.shape[4])):
        dsviewer.ivps.append(ImageViewPanel(dsviewer, dsviewer.image, dsviewer.glCanvas, dsviewer.do, chan=i))
        if dsviewer.image.data_xyztc.shape[4] > 1 and len(cmaps) > 0:
            dsviewer.do.cmaps[i] = cmaps.pop(0)
            
        dsviewer.AddPage(page=dsviewer.ivps[-1], select=True, caption=name)
        

    if dsviewer.image.data_xyztc.shape[2] > 1:
        dsviewer.AddPage(page=ArrayViewPanel(dsviewer, do=dsviewer.do, voxelsize=dsviewer.image.voxelsize), select=False, caption='Slices')

    elif dsviewer.image.data_xyztc.shape[4] > 1:
        dsviewer.civp = ColourImageViewPanel(dsviewer, dsviewer.glCanvas, dsviewer.do, dsviewer.image)
        dsviewer.civp.ivps = dsviewer.ivps
        dsviewer.AddPage(page=dsviewer.civp, select=False, caption='Composite')

    if dsviewer.image.data_xyztc.shape[2] == 1:
        # gl canvas doesn't currently work for 3D images, crashes on linux
        dsviewer._gl_im = GLImageView(dsviewer, image=dsviewer.image, glCanvas=dsviewer.glCanvas, display_opts=dsviewer.do)
        dsviewer.AddPage(page=dsviewer._gl_im, select=True, caption='GLComp')
    

