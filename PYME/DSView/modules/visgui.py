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

from PYME.Analysis.LMVis.imageView import *
from PYME.DSView.arrayViewPanel import ArrayViewPanel

class visGuiExtras:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer
        
        dsviewer.AddMenuItem("View", "E&xport Current View", self.OnExport)
        dsviewer.AddMenuItem("View", "Set as visualisation &background", self.OnViewBackground)

    def OnViewBackground(self, event):
        ivp = self.dsviewer.GetSelectedPage() #self.notebook.GetPage(self.notebook.GetSelection())

        if 'image' in dir(ivp): #is a single channel
            img = numpy.minimum(255.*(ivp.image.img - ivp.clim[0])/(ivp.clim[1] - ivp.clim[0]), 255).astype('uint8')
            self.dsviewer.glCanvas.setBackgroundImage(img, (ivp.image.imgBounds.x0, ivp.image.imgBounds.y0), pixelSize=ivp.image.pixelSize)

        self.dsviewer.glCanvas.Refresh()



    def OnExport(self, event):
        #ivp = self.notebook.GetPage(self.notebook.GetSelection())
        ivp = self.dsviewer.GetSelectedPage()
        fname = wx.FileSelector('Save Current View', default_extension='.tif', wildcard="Supported Image Files (*.tif, *.bmp, *.gif, *.jpg, *.png)|*.tif, *.bmp, *.gif, *.jpg, *.png", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

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

def Plug(dsviewer):
    dsviewer.vgextras = visGuiExtras(dsviewer)

    cmaps = [pylab.cm.r, pylab.cm.g, pylab.cm.b]

    if not 'ivps' in dir(dsviewer):
        dsviewer.ivps = []

    for name, i in zip(dsviewer.image.names, xrange(dsviewer.image.data.shape[3])):
        dsviewer.ivps.append(ImageViewPanel(dsviewer, dsviewer.image, dsviewer.glCanvas, dsviewer.do, chan=i))
        if dsviewer.image.data.shape[3] > 1 and len(cmaps) > 0:
            dsviewer.do.cmaps[i] = cmaps.pop(0)

        dsviewer.AddPage(page=dsviewer.ivps[-1], select=True, caption=name)


    if dsviewer.image.data.shape[2] > 1:
        dsviewer.AddPage(page=ArrayViewPanel(dsviewer, do=dsviewer.do), select=False, caption='Slices')

    elif dsviewer.image.data.shape[3] > 1:
        dsviewer.civp = ColourImageViewPanel(dsviewer, dsviewer.glCanvas, dsviewer.do, dsviewer.image)
        dsviewer.civp.ivps = dsviewer.ivps
        dsviewer.AddPage(page=dsviewer.civp, select=True, caption='Composite')
    

