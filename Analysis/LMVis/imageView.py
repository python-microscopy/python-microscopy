import wx

import sys,math
import numpy


class ImageViewPanel(wx.Panel):
    def __init__(self, parent, image, glCanvas):
	wx.Panel.__init__(self, parent,-1, size=parent.Size)

        self.image = image
        self.glCanvas = glCanvas

        c = self.image.img.ravel()

        #clim_upper = float(c[numpy.argsort(c)[len(c)*.95]])
        clim_upper = c.max()
        self.clim = (0, clim_upper)

        wx.EVT_PAINT(self, self.OnPaint)


    def DoPaint(self, dc):
        pixelsize = self.glCanvas.pixelsize

        centreX = (self.glCanvas.xmin + self.glCanvas.xmax)/2
        centreY = (self.glCanvas.ymin + self.glCanvas.ymax)/2
        #print centreX

        width = self.Size[0]*pixelsize
        height = self.Size[1]*pixelsize

        x0 = max(centreX  - width/2, self.image.imgBounds.x0)
        x1 = min(centreX  + width/2, self.image.imgBounds.x1)
        y0 = max(centreY  - height/2, self.image.imgBounds.y0)
        y1 = min(centreY  + height/2, self.image.imgBounds.y1)

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

        im = numpy.flipud(self.image.img[int(x0_/self.image.pixelSize):int(x1_/self.image.pixelSize):step,int(y0_/self.image.pixelSize):int(y1_/self.image.pixelSize):step ].astype('f').T)

        im = im - self.clim[0]
        im = im/(self.clim[1] - self.clim[0])

        im = (255*self.glCanvas.cmap(im)[:,:,:3]).astype('b')

        #print im.shape
            
        imw =  wx.ImageFromData(im.shape[1], im.shape[0], im.ravel())
        #print imw.Size
                                         
        imw.Rescale(imw.GetWidth()*sc,imw.GetHeight()*sc)
        #print imw.Size

        dc.Clear()
        
        dc.DrawBitmap(wx.BitmapFromImage(imw),(centreX - x1 + width/2)/pixelsize,(centreY - y1 + height/2)/pixelsize)
            
        
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

    def DoRefresh(self, event):
        pass
        


class ImageViewFrame(wx.Frame):
    def __init__(self, parent, image, glCanvas):
        wx.Frame.__init__(self, parent, -1, title='Generated Image', size=(800,800))

        self.ivp = ImageViewPanel(self, image, glCanvas)


    
