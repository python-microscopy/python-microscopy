#!/usr/bin/python

##################
# imageView.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import math
import numpy
import os
import sys
import wx
import wx.lib.agw.aui as aui
import histLimits
import pylab
import scipy.misc
import subprocess

from PYME.Analysis import thresholding

#from PYME.DSView.myviewpanel_numarray import MyViewPanel
from PYME.DSView.arrayViewPanel import ArrayViewPanel
from PYME.DSView.displayOptions import DisplayOpts
from PYME.DSView.DisplayOptionsPanel import OptionsPanel
from PYME.DSView.image import ImageStack

#from PYME.misc.auiFloatBook import AuiNotebookWithFloatingPages

class ImageViewPanel(wx.Panel):
    def __init__(self, parent, image, glCanvas, do, chan=0, zdim=0):
        wx.Panel.__init__(self, parent, -1, size=parent.Size)

        self.image = image
        self.glCanvas = glCanvas
        #self.do.zp = zp
        self.do = do
        self.chan = chan
        self.zdim = zdim
        self.cmap = pylab.cm.hot

        if len(self.image.img.shape) == 2:
            c = self.image.img.ravel()
        elif self.zdim == 0:
            c = self.image.img[self.do.zp, :,:].ravel()
        else:
            c = self.image.img[:,:,self.do.zp].ravel()

        #clim_upper = float(c[numpy.argsort(c)[len(c)*.95]])
        #clim_upper = c.max()
        #self.clim = (c.min(), clim_upper)

        
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)
    


    def DoPaint(self, dc):
        pixelsize = self.glCanvas.pixelsize

        self.centreX = (self.glCanvas.xmin + self.glCanvas.xmax)/2.
        self.centreY = (self.glCanvas.ymin + self.glCanvas.ymax)/2.
        #print centreX

        width = self.Size[0]*pixelsize
        height = self.Size[1]*pixelsize

        x0 = max(self.centreX  - width/2, self.image.imgBounds.x0)
        x1 = min(self.centreX  + width/2, self.image.imgBounds.x1)
        y0 = max(self.centreY  - height/2, self.image.imgBounds.y0)
        y1 = min(self.centreY  + height/2, self.image.imgBounds.y1)

        x0_ = x0 - self.image.imgBounds.x0
        x1_ = x1 - self.image.imgBounds.x0
        y0_ = y0 - self.image.imgBounds.y0
        y1_ = y1 - self.image.imgBounds.y0

        sc = self.image.pixelSize/pixelsize

        if sc >= 1:
            step = 1
        else:
            step = 2**(-numpy.ceil(numpy.log2(sc)))
            sc = sc*step

        #print sc

        #print (x0_/self.image.pixelSize),(x1_/self.image.pixelSize),step
        #print (y0_/self.image.pixelSize),(y1_/self.image.pixelSize),step

        if len(self.image.img.shape) == 2:
            im = numpy.flipud(self.image.img[int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step].astype('f').T)
        elif self.zdim ==0:
            im = numpy.flipud(self.image.img[self.do.zp,int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step].astype('f').T)
        else:
            im = numpy.flipud(self.image.img[int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step, self.do.zp].astype('f').T)

        im = im - self.do.Offs[self.chan] #self.clim[0]
        im = im*self.do.Gains[self.chan]    #/(self.clim[1] - self.clim[0])

        im = (255*self.do.cmaps[self.chan](im)[:,:,:3]).astype('b')

        #print im.shape
            
        imw =  wx.ImageFromData(im.shape[1], im.shape[0], im.ravel())
        #print imw.Size
                                         
        imw.Rescale(imw.GetWidth()*sc,imw.GetHeight()*sc)
        #print imw.Size
        self.curIm = imw

        dc.Clear()
        
        dc.DrawBitmap(wx.BitmapFromImage(imw),(-self.centreX + x0 + width/2)/pixelsize,(self.centreY - y1 + height/2)/pixelsize)

        #print self.glCanvas.centreCross

        if self.glCanvas.centreCross:
            print 'drawing crosshair'
            dc.SetPen(wx.Pen(wx.GREEN, 2))

            dc.DrawLine(.5*self.Size[0], 0, .5*self.Size[0], self.Size[1])
            dc.DrawLine(0, .5*self.Size[1], self.Size[0], .5*self.Size[1])

            dc.SetPen(wx.NullPen)
            
        
    def OnPaint(self,event):
        DC = wx.PaintDC(self)
        self.PrepareDC(DC)
        
        s = self.GetVirtualSize()
        MemBitmap = wx.EmptyBitmap(s.GetWidth(), s.GetHeight())
        #del DC
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)
        try:
            DC.BeginDrawing()
            
            self.DoPaint(MemDC);
            
            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            DC.EndDrawing()
        finally:
            
            del MemDC
            del MemBitmap

    def OnSize(self, event):
        self.Refresh()

    def OnWheel(self, event):
        rot = event.GetWheelRotation()

        #get translated coordinates
        xp = self.centreX + self.glCanvas.pixelsize*(event.GetX() - self.Size[0]/2)
        yp = self.centreY - self.glCanvas.pixelsize*(event.GetY() - self.Size[1]/2)

        #print xp
        #print yp
        self.glCanvas.WheelZoom(rot, xp, yp)

    def OnKeyPress(self, event):
        if event.GetKeyCode() == wx.WXK_PRIOR:
            self.do.zp =max(self.do.zp - 1, 0)
            self.Refresh()
        elif event.GetKeyCode() == wx.WXK_NEXT:
            self.do.zp = min(self.do.zp + 1, self.image.img.shape[self.zdim] - 1)
            self.Refresh()
        else:
            event.Skip()

class ColourImageViewPanel(ImageViewPanel):
    def __init__(self, parent, glCanvas, do):
        wx.Panel.__init__(self, parent, -1, size=parent.Size)

        self.ivps = []
        #self.clims = []
        #self.cmaps = []
        self.do = do

        self.glCanvas = glCanvas

        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)

    def DoPaint(self, dc):
        pixelsize = self.glCanvas.pixelsize

        self.centreX = (self.glCanvas.xmin + self.glCanvas.xmax)/2.
        self.centreY = (self.glCanvas.ymin + self.glCanvas.ymax)/2.
        #print centreX

        width = self.Size[0]*pixelsize
        height = self.Size[1]*pixelsize

        im_ = numpy.zeros((self.Size[0], self.Size[1], 3), 'uint8')

        for ivp in self.ivps:
            img = ivp.image
            #clim = ivp.clim
            cmap = ivp.do.cmaps[ivp.chan]

            x0 = max(self.centreX  - width/2, img.imgBounds.x0)
            x1 = min(self.centreX  + width/2, img.imgBounds.x1)
            y0 = max(self.centreY  - height/2, img.imgBounds.y0)
            y1 = min(self.centreY  + height/2, img.imgBounds.y1)

            x0_ = x0 - img.imgBounds.x0
            x1_ = x1 - img.imgBounds.x0
            y0_ = y0 - img.imgBounds.y0
            y1_ = y1 - img.imgBounds.y0

            sc = float(img.pixelSize/pixelsize)

            if sc >= 1:
                step = 1
            else:
                step = 2**(-numpy.ceil(numpy.log2(sc)))
                sc = sc*step

            #print sc

            #print (x0_/self.image.pixelSize),(x1_/self.image.pixelSize),step
            #print (y0_/self.image.pixelSize),(y1_/self.image.pixelSize),step

            if len(img.img.shape) == 2:
                im = numpy.flipud(img.img[int(x0_ / img.pixelSize):int(x1_ / img.pixelSize):step, int(y0_ / img.pixelSize):int(y1_ / img.pixelSize):step].astype('f').T)
            elif self.zdim ==0:
                im = numpy.flipud(img.img[self.do.zp,int(x0_ / img.pixelSize):int(x1_ / img.pixelSize):step, int(y0_ / img.pixelSize):int(y1_ / img.pixelSize):step].astype('f').T)
            else:
                im = numpy.flipud(img.img[int(x0_ / img.pixelSize):int(x1_ / img.pixelSize):step, int(y0_ / img.pixelSize):int(y1_ / img.pixelSize):step, self.do.zp].astype('f').T)

            #print clim
            #im = im.astype('f') - clim[0]
            #im = im/(clim[1] - clim[0])
            im = im - self.do.Offs[ivp.chan] #self.clim[0]
            im = im*self.do.Gains[ivp.chan]    #/(self.clim[1] - self.clim[0])

            im = numpy.minimum(im, 1)
            im = numpy.maximum(im, 0)
            
            #print im.max(), im.min()

            #print im.shape, sc

            im = scipy.misc.imresize(im, sc)

            #print im.max(), im.min()
            #print im.shape

            im = (255*cmap(im)[:,:,:3])

#            w1 = (self.glCanvas.xmax - self.glCanvas.xmin)
#            h1 = (self.glCanvas.ymax - self.glCanvas.ymin)

            #print self.centreX, self.centreY, x0, y0, width, height, pixelsize, round((-self.centreX + x0 + width/2)/pixelsize)

            dx = round((-self.centreX + x0 + width/2)/pixelsize)
            dy = round((self.centreY - y1 + height/2)/pixelsize)

            #print dx, dy, im_.shape, im_[dx:(im.shape[0] + dx), dy:(im.shape[1] + dy), :].shape
            #print self.centreX, self.centreY, x0, y0, dx, dy, width, height, pixelsize, round((-self.centreX + x0 + width/2)/pixelsize)

            im_[dy:(im.shape[0] + dy), dx:(im.shape[1] + dx), :] = im_[dy:(im.shape[0] + dy), dx:(im.shape[1] + dx), :] + im[:(im_.shape[0] - dy),:(im_.shape[1] - dx)]

            #print im.shape

        im_ = numpy.minimum(im_, 255).astype('b')
        #print im_.shape

        imw =  wx.ImageFromData(im_.shape[1], im_.shape[0], im_.ravel())
        #print imw.GetWidth()

        #imw.Rescale(imw.GetWidth()*sc,imw.GetHeight()*sc)
            #print imw.Size
        self.curIm = imw

        dc.Clear()

        #dc.DrawBitmap(wx.BitmapFromImage(imw),(-self.centreX + x0 + width/2)/pixelsize,(self.centreY - y1 + height/2)/pixelsize)
        dc.DrawBitmap(wx.BitmapFromImage(imw), 0,0)

        #print self.glCanvas.centreCross

        if self.glCanvas.centreCross:
            print 'drawing crosshair'
            dc.SetPen(wx.Pen(wx.GREEN, 2))

            dc.DrawLine(.5*self.Size[0], 0, .5*self.Size[0], self.Size[1])
            dc.DrawLine(0, .5*self.Size[1], self.Size[0], .5*self.Size[1])

            dc.SetPen(wx.NullPen)
        

class ImageViewFrame(wx.Frame):
    def __init__(self, parent, image, glCanvas, title='Generated Image',zp=0, zdim=0):
        wx.Frame.__init__(self, parent, -1, title=title, size=(800,800))

        self.ivp = ImageViewPanel(self, image, glCanvas, zp=zp, zdim=zdim)
        self.parent = parent
        
        self.hlCLim = None
        self.cmn = 'hot'

        self.SetMenuBar(self.CreateMenuBar())
        wx.EVT_CLOSE(self, self.OnClose)

    def CreateMenuBar(self):
        
        # Make a menubar
        file_menu = wx.Menu()

        #ID_SAVE = wx.NewId()
        #ID_CLOSE = wx.NewId()
        ID_EXPORT = wx.NewId()

        ID_VIEW_COLOURLIM = wx.NewId()
        self.ID_VIEW_CMAP_INVERT = wx.NewId()
        
        file_menu.Append(wx.ID_SAVE, "&Save")
        file_menu.Append(ID_EXPORT, "E&xport Current View")
        
        file_menu.AppendSeparator()
        
        file_menu.Append(wx.ID_CLOSE, "&Close")

        view_menu = wx.Menu()
        view_menu.AppendCheckItem(ID_VIEW_COLOURLIM, "&Colour Scaling")

        self.cmap_menu = wx.Menu()

        self.cmap_menu.AppendCheckItem(self.ID_VIEW_CMAP_INVERT, "&Invert")
        self.cmap_menu.AppendSeparator()

        cmapnames = pylab.cm.cmapnames
        cmapnames.sort()
        self.cmapIDs = {}
        for cmn in cmapnames:
            cmmId = wx.NewId()
            self.cmapIDs[cmmId] = cmn
            self.cmap_menu.AppendRadioItem(cmmId, cmn)
            if cmn == self.ivp.cmap.name:
                self.cmap_menu.Check(cmmId, True)
            self.Bind(wx.EVT_MENU, self.OnChangeLUT, id=cmmId)

        view_menu.AppendMenu(-1,'&LUT', self.cmap_menu)

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(view_menu, "&View")

        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_MENU, self.OnExport, id=ID_EXPORT)
        self.Bind(wx.EVT_MENU, self.OnViewCLim, id=ID_VIEW_COLOURLIM)
        self.Bind(wx.EVT_MENU, self.OnCMapInvert, id=self.ID_VIEW_CMAP_INVERT)

        return menu_bar


    def OnSave(self, event):
        fname = wx.FileSelector('Save Image ...', default_extension='.tif', wildcard="TIFF files (*.tif)|*.tif", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            self.ivp.image.save(fname)

        self.SetTitle(fname)


    def OnClose(self, event):
        self.parent.generatedImages.remove(self)
        self.Destroy()

    def OnExport(self, event):
        fname = wx.FileSelector('Save Current View', default_extension='.tif', wildcard="Supported Image Files (*.tif, *.bmp, *.gif, *.jpg, *.png)|*.tif, *.bmp, *.gif, *.jpg, *.png", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            ext = os.path.splitext(fname)[-1]
            if ext == '.tif':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_TIF)
            elif ext == '.png':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_PNG)
            elif ext == '.jpg':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_JPG)
            elif ext == '.gif':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_GIF)
            elif ext == '.bmp':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_BMP)

    def OnViewCLim(self, event):
        if self.hlCLim == None:
            if len(self.ivp.image.img.shape) == 2:
                c = self.ivp.image.img.ravel()
            else:
                c = self.ivp.image.img[self.ivp.zp, :,:].ravel()
            #ID = histLimits.ShowHistLimitFrame(self, 'Colour Scaling', c, self.ivp.clim[0], self.ivp.clim[1], size=(200, 100), log=True)
            self.hlCLim = histLimits.HistLimitPanel(self, -1, c, self.ivp.clim[0], self.ivp.clim[1], size=(200, 100), pos=(self.Size[0] - 200, 0), log=True)

            self.hlCLim.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnCLimChanged)
        else:
            self.hlCLim.Destroy()
            self.hlCLim = None

    def OnCLimChanged(self, event):
        self.ivp.clim = (event.lower, event.upper)
        self.ivp.Refresh()

    def OnChangeLUT(self, event):
        #print event
        self.cmn = self.cmapIDs[event.GetId()]
        cmn = self.cmn
        if self.cmap_menu.IsChecked(self.ID_VIEW_CMAP_INVERT):
            cmn = cmn + '_r'
        self.ivp.cmap = pylab.cm.__getattribute__(cmn)
        self.ivp.Refresh()

    def OnCMapInvert(self, event):
        cmn = self.cmn
        if self.cmap_menu.IsChecked(self.ID_VIEW_CMAP_INVERT):
            cmn = cmn + '_r'
        self.ivp.cmap = pylab.cm.__getattribute__(cmn)
        self.ivp.Refresh()

class ColourImageViewFrame(wx.Frame):
    def __init__(self, parent, glCanvas, title='Generated Image',zp=0, zdim=0):
        wx.Frame.__init__(self, parent, -1, title=title, size=(800,800))

        self.ivp = ColourImageViewPanel(self, glCanvas)
        self.parent = parent

        #self.SetMenuBar(self.CreateMenuBar())
        wx.EVT_CLOSE(self, self.OnClose)

    def CreateMenuBar(self):

        # Make a menubar
        file_menu = wx.Menu()

        #ID_SAVE = wx.NewId()
        #ID_CLOSE = wx.NewId()
        ID_EXPORT = wx.NewId()

        ID_VIEW_COLOURLIM = wx.NewId()
        self.ID_VIEW_CMAP_INVERT = wx.NewId()

        #file_menu.Append(wx.ID_SAVE, "&Save")
        file_menu.Append(ID_EXPORT, "E&xport Current View")

        file_menu.AppendSeparator()

        file_menu.Append(wx.ID_CLOSE, "&Close")

        view_menu = wx.Menu()
        view_menu.AppendCheckItem(ID_VIEW_COLOURLIM, "&Colour Scaling")

        self.cmap_menu = wx.Menu()

        self.cmap_menu.AppendCheckItem(self.ID_VIEW_CMAP_INVERT, "&Invert")
        self.cmap_menu.AppendSeparator()

        cmapnames = pylab.cm.cmapnames
        cmapnames.sort()
        self.cmapIDs = {}
        for cmn in cmapnames:
            cmmId = wx.NewId()
            self.cmapIDs[cmmId] = cmn
            self.cmap_menu.AppendRadioItem(cmmId, cmn)
            if cmn == self.ivp.cmap.name:
                self.cmap_menu.Check(cmmId, True)
            self.Bind(wx.EVT_MENU, self.OnChangeLUT, id=cmmId)

        view_menu.AppendMenu(-1,'&LUT', self.cmap_menu)

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(view_menu, "&View")

        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_MENU, self.OnExport, id=ID_EXPORT)
        self.Bind(wx.EVT_MENU, self.OnViewCLim, id=ID_VIEW_COLOURLIM)
        self.Bind(wx.EVT_MENU, self.OnCMapInvert, id=self.ID_VIEW_CMAP_INVERT)

        return menu_bar


    def OnSave(self, event):
        fname = wx.FileSelector('Save Image ...', default_extension='.tif', wildcard="TIFF files (*.tif)|*.tif", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            self.ivp.image.save(fname)

        self.SetTitle(fname)


    def OnClose(self, event):
        #self.parent.generatedImages.remove(self)
        self.Destroy()

    def OnExport(self, event):
        fname = wx.FileSelector('Save Current View', default_extension='.tif', wildcard="Supported Image Files (*.tif, *.bmp, *.gif, *.jpg, *.png)|*.tif, *.bmp, *.gif, *.jpg, *.png", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            ext = os.path.splitext(fname)[-1]
            if ext == '.tif':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_TIF)
            elif ext == '.png':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_PNG)
            elif ext == '.jpg':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_JPG)
            elif ext == '.gif':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_GIF)
            elif ext == '.bmp':
                self.ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_BMP)

    def OnViewCLim(self, event):
        if self.hlCLim == None:
            if len(self.ivp.image.img.shape) == 2:
                c = self.ivp.image.img.ravel()
            else:
                c = self.ivp.image.img[self.ivp.zp, :,:].ravel()
            #ID = histLimits.ShowHistLimitFrame(self, 'Colour Scaling', c, self.ivp.clim[0], self.ivp.clim[1], size=(200, 100), log=True)
            self.hlCLim = histLimits.HistLimitPanel(self, -1, c, self.ivp.clim[0], self.ivp.clim[1], size=(200, 100), pos=(self.Size[0] - 200, 0), log=True)

            self.hlCLim.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnCLimChanged)
        else:
            self.hlCLim.Destroy()
            self.hlCLim = None

    def OnCLimChanged(self, event):
        self.ivp.clim = (event.lower, event.upper)
        self.ivp.Refresh()

    def OnChangeLUT(self, event):
        #print event
        self.cmn = self.cmapIDs[event.GetId()]
        cmn = self.cmn
        if self.cmap_menu.IsChecked(self.ID_VIEW_CMAP_INVERT):
            cmn = cmn + '_r'
        self.ivp.cmap = pylab.cm.__getattribute__(cmn)
        self.ivp.Refresh()

    def OnCMapInvert(self, event):
        cmn = self.cmn
        if self.cmap_menu.IsChecked(self.ID_VIEW_CMAP_INVERT):
            cmn = cmn + '_r'
        self.ivp.cmap = pylab.cm.__getattribute__(cmn)
        self.ivp.Refresh()
        
from PYME.Acquire.MetaDataHandler import NestedClassMDHandler
from PYME.DSView import modules as dsvmods

class MultiChannelImageViewFrame(wx.Frame):
    def __init__(self, parent, glCanvas, images, names=['Image'], title='Generated Image',zp=0, zdim=2):
        wx.Frame.__init__(self, parent, -1, title=title, size=(800,800))

        self.glCanvas = glCanvas
        self.parent = parent
        
        self.images = images
        self.names = [n or 'Image' for n in names]
        self.frame = self

        
        md = NestedClassMDHandler()
        md.setEntry('voxelsize.x', images[0].pixelSize)
        md.setEntry('voxelsize.y', images[0].pixelSize)
        md.setEntry('voxelsize.z', images[0].sliceSize)

        md.setEntry('ChannelNames', self.names)

        self.image = ImageStack([numpy.atleast_3d(im.img) for im in images], md)

        self.paneHooks = []
        self.updateHooks = []
        self.statusHooks = []
        self.installedModules = []

        #self.notebook = AuiNotebookWithFloatingPages(id=-1, parent=self, style=wx.aui.AUI_NB_TAB_SPLIT)
        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT | aui.AUI_MGR_AUTONB_NO_CAPTION)
        atabstyle = self._mgr.GetAutoNotebookStyle()
        self._mgr.SetAutoNotebookStyle((atabstyle ^ aui.AUI_NB_BOTTOM) | aui.AUI_NB_TOP)
        # tell AuiManager to manage this frame
        self._mgr.SetManagedWindow(self)

        self.ivps = []
        self.pane0 = None

        asp = self.images[0].sliceSize/self.images[0].pixelSize
        if asp == 0:
            asp = 1
        ims = self.images[0].img.shape

        if len(ims) == 2:
            ims = list(ims) + [1]

        self.do = DisplayOpts([img.img.reshape(ims) for img in self.images], asp)
        self.do.Optimise()
        self.do.zp = zp
        self.do.names = self.names

        cmaps = [pylab.cm.r, pylab.cm.g, pylab.cm.b]

        for img, name, i in zip(self.images, self.names, xrange(len(self.images))):
            self.ivps.append(ImageViewPanel(self, img, glCanvas, self.do, chan=i, zdim=zdim))
            if len(self.images) > 1 and len(cmaps) > 0:
                self.do.cmaps[i] = cmaps.pop(0)

            self.AddPage(page=self.ivps[-1], select=True, caption=name)


        if len(img.img.shape) > 2:
            asp = self.images[0].sliceSize/self.images[0].pixelSize
            if asp == 0:
                asp = 1
            
            self.AddPage(page=ArrayViewPanel(self, do=self.do, aspect = asp), select=False, caption='Slices')
            
        elif len(self.images) > 1:
            self.civp = ColourImageViewPanel(self, glCanvas, self.do)
            self.civp.ivps = self.ivps
            self.AddPage(page=self.civp, select=True, caption='Composite')

        

        self.menubar = self.CreateMenuBar()
        self.SetMenuBar(self.menubar)

        self.limitsFrame = None

        self.optionspanel = OptionsPanel(self, self.do, thresholdControls=True)
        self.optionspanel.SetSize(self.optionspanel.GetBestSize())
        pinfo = aui.AuiPaneInfo().Name("optionsPanel").Right().Caption('Display Settings').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        self._mgr.AddPane(self.optionspanel, pinfo)

        if self.do.ds.shape[2] > 1:
            from PYME.DSView.modules import playback
            self.playbackpanel = playback.PlayPanel(self, self)
            self.playbackpanel.SetSize(self.playbackpanel.GetBestSize())

            pinfo1 = aui.AuiPaneInfo().Name("playbackPanel").Bottom().Caption('Playback').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
            self._mgr.AddPane(self.playbackpanel, pinfo1)
            self.do.WantChangeNotification.append(self.playbackpanel.update)

        dsvmods.loadMode('visGUI', self)
        self.CreateModuleMenu()

        self._mgr.Update()
        self._mgr.MinimizePane(pinfo)

        wx.EVT_CLOSE(self, self.OnClose)
        self.do.WantChangeNotification.append(self.Refresh)

    def CreateModuleMenu(self):
        self.modMenuIds = {}
        self.mModules = wx.Menu()
        for mn in dsvmods.allmodules:
            id = wx.NewId()
            self.mModules.AppendCheckItem(id, mn)
            self.modMenuIds[id] = mn
            if mn in self.installedModules:
                self.mModules.Check(id, True)

            wx.EVT_MENU(self, id, self.OnToggleModule)

        self.menubar.Append(self.mModules, "&Modules")

    def OnToggleModule(self, event):
        id = event.GetId()
        mn = self.modMenuIds[id]
        if self.mModules.IsChecked(id):
            dsvmods.loadModule(mn, self)

        if mn in self.installedModules:
            self.mModules.Check(id, True)

        #self.CreateFoldPanel()
        self._mgr.Update()



    def AddPage(self, page=None, select=True,caption='Dummy'):
        if self.pane0 == None:
            name = caption.replace(' ', '')
            self._mgr.AddPane(page, aui.AuiPaneInfo().
                          Name(name).Caption(caption).Centre().CloseButton(False))
            self.pane0 = name

        else:
            self._mgr.Update()
            pn = self._mgr.GetPaneByName(self.pane0)
            if pn.IsNotebookPage():
                print pn.notebook_id
                nbs = self._mgr.GetNotebooks()
                if len(nbs) > pn.notebook_id:
                    currPage = nbs[pn.notebook_id].GetSelection()
                self._mgr.AddPane(page, aui.AuiPaneInfo().
                              Name(caption.replace(' ', '')).Caption(caption).CloseButton(False).NotebookPage(pn.notebook_id))
                if (not select) and len(nbs) > pn.notebook_id:
                    nbs[pn.notebook_id].SetSelection(currPage)
            else:
                self._mgr.AddPane(page, aui.AuiPaneInfo().
                              Name(caption.replace(' ', '')).Caption(caption).CloseButton(False), target=pn)
                #nb = self._mgr.GetNotebooks()[0]
                #if not select:
                #    nb.SetSelection(0)

        self._mgr.Update()

    def update(self):
        #if 'playbackpanel' in dir(self):
        #        self.playbackpanel.update()
        self.Refresh()

    def GetSelectedPage(self):
        nbs = self._mgr.GetNotebooks()
        currPage = nbs[0].GetCurrentPage()

        return currPage


    def CreateMenuBar(self):

        # Make a menubar
        file_menu = wx.Menu()

        #ID_SAVE = wx.NewId()
        #ID_CLOSE = wx.NewId()
        ID_EXPORT = wx.NewId()
        #ID_SAVEALL = wx.NewId()

        #ID_VIEW_CONSOLE = wx.NewId()
        ID_VIEW_BACKGROUND = wx.NewId()
        ID_FILTER_GAUSS = wx.NewId()
        #ID_COLOC = wx.NewId()
        #ID_3D_ISOSURF = wx.NewId()
        #ID_3D_VOLUME = wx.NewId()
        self.ID_VIEW_CMAP_INVERT = wx.NewId()

        file_menu.Append(wx.ID_SAVE, "&Save Channel")
        #file_menu.Append(ID_SAVEALL, "Save &Multi-channel")
        file_menu.Append(ID_EXPORT, "E&xport Current View")

        file_menu.AppendSeparator()

        file_menu.Append(wx.ID_CLOSE, "&Close")

        view_menu = wx.Menu()
        #view_menu.Append(ID_VIEW_CONSOLE, "&Console")
        view_menu.Append(ID_VIEW_BACKGROUND, "Set as visualisation &background")

        self.mProcessing = wx.Menu()
        self.mProcessing.Append(ID_FILTER_GAUSS, "&Gaussian Filter")
        #self.mProcessing.Append(ID_COLOC, "&Colocalisation")


        #td_menu = wx.Menu()
        #td_menu.Append(ID_3D_ISOSURF, "&Isosurface")
        #td_menu.Append(ID_3D_VOLUME, "&Volume")

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(view_menu, "&View")
        menu_bar.Append(self.mProcessing, "&Processing")
        #menu_bar.Append(td_menu, "&3D")

        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        #self.Bind(wx.EVT_MENU, self.OnSaveChannels, id=ID_SAVEALL)
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_MENU, self.OnExport, id=ID_EXPORT)
        #self.Bind(wx.EVT_MENU, self.OnViewCLim, id=ID_VIEW_COLOURLIM)
        #self.Bind(wx.EVT_MENU, self.OnViewConsole, id=ID_VIEW_CONSOLE)
        self.Bind(wx.EVT_MENU, self.OnViewBackground, id=ID_VIEW_BACKGROUND)
        #self.Bind(wx.EVT_MENU, self.On3DIsosurf, id=ID_3D_ISOSURF)
        #self.Bind(wx.EVT_MENU, self.On3DVolume, id=ID_3D_VOLUME)
        self.Bind(wx.EVT_MENU, self.OnGaussianFilter, id=ID_FILTER_GAUSS)
        #self.Bind(wx.EVT_MENU, self.OnColoc, id=ID_COLOC)
        #self.Bind(wx.EVT_MENU, self.OnCMapInvert, id=self.ID_VIEW_CMAP_INVERT)

        return menu_bar


    def OnSave(self, event):
        self.image.save()
#        ivp = self.GetSelectedPage()# self.notebook.GetPage(self.notebook.GetSelection())
#
#        if 'image' in dir(ivp): #is a single channel
#            fname = wx.FileSelector('Save Image ...', default_extension='.tif', wildcard="TIFF files (*.tif)|*.tif", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
#
#            if not fname == "":
#                ivp.image.save(fname)
#
#                n = self.names[self.ivps.index(ivp)]
#                #self.notebook.SetPageText(self.notebook.GetSelection(), n + ' - ' + os.path.split(fname)[-1])
#                self._mgr.GetPaneByWidget(ivp).Caption(n + ' - ' + os.path.split(fname)[-1])
#                self._mgr.Update()
#
#        else:
#            #wx.MessageBox('Saving composites not supported yet')
#            self.OnSaveChannels(None)

    def OnViewBackground(self, event):
        ivp = self.GetSelectedPage() #self.notebook.GetPage(self.notebook.GetSelection())

        if 'image' in dir(ivp): #is a single channel
            img = numpy.minimum(255.*(ivp.image.img - ivp.clim[0])/(ivp.clim[1] - ivp.clim[0]), 255).astype('uint8')
            self.glCanvas.setBackgroundImage(img, (ivp.image.imgBounds.x0, ivp.image.imgBounds.y0), pixelSize=ivp.image.pixelSize)

        self.glCanvas.Refresh()

    #def OnViewConsole(self, event):
    #    from PYME.DSView.modules import shell
    #    shell.Plug(self)


    def OnSaveChannels(self, event):
        fname = wx.FileSelector('Save Image ...', default_extension='.tif', wildcard="TIFF files (*.tif)|*.tif", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            img = numpy.array([im.img.astype('f') for im in self.images])

            from PYME.FileUtils import saveTiffStack

            saveTiffStack.saveTiffMultipage(img, fname)



    def OnClose(self, event):
        if self in self.parent.generatedImages:
            self.parent.generatedImages.remove(self)
        self.Destroy()

    def OnExport(self, event):
        ivp = self.notebook.GetPage(self.notebook.GetSelection())
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

    #def OnViewCLim(self, event):
    #    if self.limitsFrame == None:
    #        px, py = self.GetPosition()
    #        self.limitsFrame = DispSettingsFrame(self, -1, pos=(px+self.Size[0], py))
    #        self.limitsFrame.Show()
    #    else:
    #        self.limitsFrame.Destroy()
    #        self.limitsFrame = None

#    def On3DIsosurf(self, event):
#        from enthought.mayavi import mlab
#
#        f = mlab.figure()
#
#        #asp = self.images[0].sliceSize/self.images[0].pixelSize
#        #asp = self.image.mdh.getEntry('voxelsize.z')/
#        #if asp == 0:
#        #    asp = 1.
#
#        for i in range(self.image.data.shape[3]):
#            c = mlab.contour3d(self.image.data[:,:,:,i], contours=[self.do.Offs[i] + .5/self.do.Gains[i]], color = pylab.cm.gist_rainbow(float(i)/len(self.images))[:3])
#            c.mlab_source.dataset.spacing = (self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z'))
#
#    def On3DVolume(self, event):
#        from enthought.mayavi import mlab
#
#        f = mlab.figure()
#
#        #asp = self.images[0].sliceSize/self.images[0].pixelSize
#        #if asp == 0:
#        #    asp = 1.
#
#        #for im, ivp, i in zip(self.images, self.ivps, range(len(self.images))):
#        for i in range(self.image.data.shape[3]):
#            #c = mlab.contour3d(im.img, contours=[pylab.mean(ivp.clim)], color = pylab.cm.gist_rainbow(float(i)/len(self.images))[:3])
#            v = mlab.pipeline.volume(mlab.pipeline.scalar_field(numpy.minimum(255*(self.image.data[:,:,:,i] -self.do.Offs[i])*self.do.Gains[i], 254).astype('uint8')))
#            v.volume.scale = (self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z'))

    def OnGaussianFilter(self, event):
        from scipy.ndimage import gaussian_filter
        from PYME.Analysis.LMVis.visHelpers import ImageBounds, GeneratedImage

        dlg = wx.TextEntryDialog(self, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')

        if dlg.ShowModal() == wx.ID_OK:
            sigmas = eval(dlg.GetValue())
            #print sigmas
            #print self.images[0].img.shape
            filt_ims = [GeneratedImage(gaussian_filter(im.img, sigmas), im.imgBounds, im.pixelSize, im.sliceSize) for im in self.images]

            imfc = MultiChannelImageViewFrame(self.parent, self.parent.glCanvas, filt_ims, self.names, title='Filtered Image - %3.1fnm bins' % self.images[0].pixelSize)

            self.parent.generatedImages.append(imfc)
            imfc.Show()

        dlg.Destroy()

#    def OnColoc(self, event):
#        from PYME.Analysis.Colocalisation import correlationCoeffs, edtColoc
#
#        #assume we have exactly 2 channels #FIXME - add a selector
#        #grab image data
#        #imA = self.images[0].img
#        #imB = self.images[1].img
#        imA = self.image.data[:,:,:,0].squeeze()
#        imB = self.image.data[:,:,:,1].squeeze()
#
#        #assume threshold is half the colour bounds - good if using threshold mode
#        tA = self.do.Offs[0] + .5/self.do.Gains[0] #pylab.mean(self.ivps[0].clim)
#        tB = self.do.Offs[1] + .5/self.do.Gains[1] #pylab.mean(self.ivps[0].clim)
#
#        try:
#            nameA, nameB = self.image.mdh.getEntry('ChannelNames')[:2]
#        except:
#            nameA = 'Channel 1'
#            nameB = 'Channel 2'
#
#        #nameA = self.names[0]
#        #nameB = self.names[1]
#
#        #voxelsize = [self.images[0].pixelSize, self.images[0].pixelSize, self.images[0].sliceSize]
#        voxelsize = [self.image.mdh.getEntry('voxelsize.x') ,self.image.mdh.getEntry('voxelsize.y'), self.image.mdh.getEntry('voxelsize.z')]
#        voxelsize = voxelsize[:imA.ndim] #trunctate to number of dimensions
#
#        pearson = correlationCoeffs.pearson(imA, imB)
#        MA, MB = correlationCoeffs.thresholdedManders(imA, imB, tA, tB)
#
#        bnA, bmA, binsA = edtColoc.imageDensityAtDistance(imB, imA > tA, voxelsize)
#        bnB, bmB, binsB = edtColoc.imageDensityAtDistance(imA, imB > tB, voxelsize)
#
#        pylab.figure()
#        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
#        pylab.subplot(211)
#        pylab.bar(binsA[:-1], bmA, binsA[1] - binsA[0])
#        pylab.xlabel('Distance from edge of %s [nm]' % nameA)
#        pylab.ylabel('Density of %s' % nameB)
#
#        pylab.subplot(212)
#        pylab.bar(binsB[:-1], bmB, binsB[1] - binsB[0])
#        pylab.xlabel('Distance from edge of %s [nm]' % nameB)
#        pylab.ylabel('Density of %s' % nameA)
#
#        pylab.figure()
#        pylab.figtext(.1, .95, 'Pearson: %2.2f   M1: %2.2f M2: %2.2f' % (pearson, MA, MB))
#        pylab.subplot(211)
#        pylab.bar(binsA[:-1], bmA*bnA, binsA[1] - binsA[0])
#        pylab.xlabel('Distance from edge of %s [nm]' % nameA)
#        pylab.ylabel('Fraction of %s' % nameB)
#
#        pylab.subplot(212)
#        pylab.bar(binsB[:-1], bmB*bnB, binsB[1] - binsB[0])
#        pylab.xlabel('Distance from edge of %s [nm]' % nameB)
#        pylab.ylabel('Fraction of %s' % nameA)
#
#
#

    def GetChannel(self, chan):
        if not type(chan) == int:
            chan = self.names.index(chan)

        return self.images[chan].img

    def GetChannelMask(self, chan):
        if not type(chan) == int:
            chan = self.names.index(chan)

        return self.images[chan].img > numpy.mean(self.ivps[chan].clim)

    def GetChannelVoxSize(self, chan):
        if not type(chan) == int:
            chan = self.names.index(chan)

        if self.images[chan].img.ndim == 2:
            return self.images[chan].img.pixelSize, self.images[chan].pixelSize
        else:
            #3D
            return self.images[chan].pixelSize, self.images[chan].pixelSize, self.images[chan].sliceSize



class DispSettingsPanel(wx.Panel):
    def __init__(self, parent, **kwargs):
        wx.Panel.__init__(self, parent, -1, *kwargs)

        self.parent = parent

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.hIds = []
        self.cIds = []
        self.cbIds = []
        self.hClims = []

        cmapnames = pylab.cm.cmapnames# + [n + '_r' for n in pylab.cm.cmapnames]
        #cmapnames.sort()
        
        for ivp, cn in zip(self.parent.ivps, self.parent.names):
            ssizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, cn), wx.VERTICAL)

            id = wx.NewId()
            self.hIds.append(id)
            c = ivp.image.img.ravel()
            hClim = histLimits.HistLimitPanel(self, id, c[::(len(c)/1e4)], ivp.clim[0], ivp.clim[1], size=(150, 80), log=True)
            self.hClims.append(hClim)

            hClim.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnCLimChanged)

            ssizer.Add(hClim, 0, wx.ALL, 5)

            id = wx.NewId()
            self.cIds.append(id)
            cCmap = wx.Choice(self, id, choices=cmapnames)
            cCmap.SetSelection(cmapnames.index(ivp.cmap.name))
            cCmap.Bind(wx.EVT_CHOICE, self.OnCMapChanged)
            ssizer.Add(cCmap, 0, wx.ALL, 5)

            vsizer.Add(ssizer, 0, wx.ALL, 5)

        ssizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Segmentation'), wx.VERTICAL)

        self.cbShowThreshold = wx.CheckBox(self, -1, 'Threshold mode')
        self.cbShowThreshold.Bind(wx.EVT_CHECKBOX, self.OnShowThreshold)
        ssizer.Add(self.cbShowThreshold, 0, wx.ALL, 5)

        self.bIsodataThresh = wx.Button(self, -1, 'Isodata')
        self.bIsodataThresh.Bind(wx.EVT_BUTTON, self.OnIsodataThresh)
        self.bIsodataThresh.Enable(False)
        ssizer.Add(self.bIsodataThresh, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)


        hsizer=wx.BoxSizer(wx.HORIZONTAL)
        self.tPercThresh = wx.TextCtrl(self, -1, '.80', size=[30, -1])
        self.tPercThresh.Enable(False)
        hsizer.Add(self.tPercThresh, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 1)
        self.bPercThresh = wx.Button(self, -1, 'Signal Fraction')
        self.bPercThresh.Bind(wx.EVT_BUTTON, self.OnSignalFracThresh)
        self.bPercThresh.Enable(False)
        hsizer.Add(self.bPercThresh, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 1)
        ssizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)

        vsizer.Add(ssizer, 0, wx.ALL|wx.EXPAND, 5)

        self.SetSizerAndFit(vsizer)

    def OnShowThreshold(self, event):
        tMode = self.cbShowThreshold.GetValue()
        for hClim in self.hClims:
            hClim.SetThresholdMode(tMode)

        self.bIsodataThresh.Enable(tMode)
        self.tPercThresh.Enable(tMode)
        self.bPercThresh.Enable(tMode)


    def OnIsodataThresh(self, event):
        for ivp, hClim in zip(self.parent.ivps, self.hClims):
            t = thresholding.isodata_f(ivp.image.img)
            hClim.SetValueAndFire((t,t))

    def OnSignalFracThresh(self, event):
        frac = max(0., min(1., float(self.tPercThresh.GetValue())))
        for ivp, hClim in zip(self.parent.ivps, self.hClims):
            t = thresholding.signalFraction(ivp.image.img, frac)
            hClim.SetValueAndFire((t,t))

    def OnCLimChanged(self, event):
        #print event.GetId()
        ind = self.hIds.index(event.GetId())
        self.parent.ivps[ind].clim = (event.lower, event.upper)
        self.parent.Refresh()

    def OnCMapChanged(self, event):
        #print event.GetId()
        ind = self.cIds.index(event.GetId())

        cmn = event.GetString()

#        if self.cmap_menu.IsChecked(self.cbIds[ind]):
#            cmn = cmn + '_r'
            
        self.parent.ivps[ind].cmap = pylab.cm.__getattribute__(cmn)
        self.parent.Refresh()







    
