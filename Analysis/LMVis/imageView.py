import math
import numpy
import os
import sys
import wx
import histLimits

class ImageViewPanel(wx.Panel):
    def __init__(self, parent, image, glCanvas, zp=0):
        wx.Panel.__init__(self, parent, -1, size=parent.Size)

        self.image = image
        self.glCanvas = glCanvas
        self.zp = zp

        if len(self.image.img.shape) == 2:
            c = self.image.img.ravel()
        else:
            c = self.image.img[self.zp, :,:].ravel()

        #clim_upper = float(c[numpy.argsort(c)[len(c)*.95]])
        clim_upper = c.max()
        self.clim = (c.min(), clim_upper)

        
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)
    


    def DoPaint(self, dc):
        pixelsize = self.glCanvas.pixelsize

        self.centreX = (self.glCanvas.xmin + self.glCanvas.xmax)/2
        self.centreY = (self.glCanvas.ymin + self.glCanvas.ymax)/2
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
        else:
            im = numpy.flipud(self.image.img[self.zp,int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step].astype('f').T)

        im = im - self.clim[0]
        im = im/(self.clim[1] - self.clim[0])

        im = (255*self.glCanvas.cmap(im)[:,:,:3]).astype('b')

        #print im.shape
            
        imw =  wx.ImageFromData(im.shape[1], im.shape[0], im.ravel())
        #print imw.Size
                                         
        imw.Rescale(imw.GetWidth()*sc,imw.GetHeight()*sc)
        #print imw.Size
        self.curIm = imw

        dc.Clear()
        
        dc.DrawBitmap(wx.BitmapFromImage(imw),(self.centreX - x1 + width/2)/pixelsize,(self.centreY - y1 + height/2)/pixelsize)
            
        
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
            self.zp =(self.zp - 1)
            self.Refresh()
        elif event.GetKeyCode() == wx.WXK_NEXT:
            self.zp = (self.zp + 1)
            self.Refresh()
        else:
            event.Skip()

        

class ImageViewFrame(wx.Frame):
    def __init__(self, parent, image, glCanvas, title='Generated Image',zp=0):
        wx.Frame.__init__(self, parent, -1, title=title, size=(800,800))

        self.ivp = ImageViewPanel(self, image, glCanvas, zp=zp)
        self.parent = parent
        
        self.hlCLim = None

        self.SetMenuBar(self.CreateMenuBar())
        wx.EVT_CLOSE(self, self.OnClose)

    def CreateMenuBar(self):
        
        # Make a menubar
        file_menu = wx.Menu()

        #ID_SAVE = wx.NewId()
        #ID_CLOSE = wx.NewId()
        ID_EXPORT = wx.NewId()

        ID_VIEW_COLOURLIM = wx.NewId()
        
        file_menu.Append(wx.ID_SAVE, "&Save")
        file_menu.Append(ID_EXPORT, "E&xport Current View")
        
        file_menu.AppendSeparator()
        
        file_menu.Append(wx.ID_CLOSE, "&Close")

        view_menu = wx.Menu()
        view_menu.AppendCheckItem(ID_VIEW_COLOURLIM, "&Colour Scaling")

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(view_menu, "&View")

        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        self.Bind(wx.EVT_MENU, self.OnExport, id=ID_EXPORT)
        self.Bind(wx.EVT_MENU, self.OnViewCLim, id=ID_VIEW_COLOURLIM)

        return menu_bar


    def OnSave(self, event):
        fname = wx.FileSelector('Save Image ...', default_extension='.tif', wildcard="TIFF files (*.tif)|*.tif", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)

        if not fname == "":
            self.ivp.image.save(fname)


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

    
