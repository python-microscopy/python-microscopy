#!/usr/bin/python
##################
# visgui.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.Analysis.LMVis.imageView import *
from PYME.DSView.arrayViewPanel import ArrayViewPanel

class visGuiExtras:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        ID_EXPORT_VIEW = wx.NewId()
        ID_VIEW_BACKGROUND = wx.NewId()

        dsviewer.view_menu.Append(ID_EXPORT_VIEW, "E&xport Current View")
        dsviewer.view_menu.Append(ID_VIEW_BACKGROUND, "Set as visualisation &background")

        dsviewer.Bind(wx.EVT_MENU, self.OnExport, id=ID_EXPORT_VIEW)
        dsviewer.Bind(wx.EVT_MENU, self.OnViewBackground, id=ID_VIEW_BACKGROUND)

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
        dsviewer.AddPage(page=ArrayViewPanel(dsviewer, do=dsviewer.do, aspect = asp), select=False, caption='Slices')

    elif dsviewer.image.data.shape[3] > 1:
        dsviewer.civp = ColourImageViewPanel(dsviewer, dsviewer.glCanvas, dsviewer.do, dsviewer.image)
        dsviewer.civp.ivps = dsviewer.ivps
        dsviewer.AddPage(page=dsviewer.civp, select=True, caption='Composite')
    

