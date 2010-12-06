#!/usr/bin/python
##################
# renderers.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
from PYME.Analysis.LMVis.visHelpers import ImageBounds, GeneratedImage
from PYME.Analysis.LMVis import genImageDialog
from PYME.Analysis.LMVis import imageView

class CurrentRenderer:
    '''Renders current view (in black and white). Only renderer not to take care
    of colour channels. Simplest renderer and as such also the base class for all 
    others'''

    name = 'Current'
    mode = 'current'
    
    def __init__(self, visFr):
        self.visFr = visFr

        self._addMenuItems()

    def Generate(self, event=None):
        dlg = genImageDialog.GenImageDialog(self.visFr, mode=self.mode)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()

            bCurr = wx.BusyCursor()

            im = self.genIm(dlg)

            x0 = max(self.visFr.glCanvas.xmin, self.visFr.imageBounds.x0)
            y0 = max(self.visFr.glCanvas.ymin, self.visFr.imageBounds.y0)
            x1 = min(self.visFr.glCanvas.xmax, self.visFr.imageBounds.x1)
            y1 = min(self.visFr.glCanvas.ymax, self.visFr.imageBounds.y1)

            #imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)
            imb = ImageBounds(x0, y0, x1, y1)

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.visFr.glCanvas)
            #self.generatedImages.append(imf)
            imf.Show()

            self.visFr.RefreshView()

        dlg.Destroy()
        return imf

    def genIm(self, dlg):
        oldcmap = self.visFr.glCanvas.cmap
        self.visFr.glCanvas.setCMap(pylab.cm.gray)
        im = self.visFr.glCanvas.getIm(dlg.getPixelSize())

        self.visFr.glCanvas.setCMap(oldcmap)

        return im

class ColourRenderer(CurrentRenderer):
    '''Base class for all other renderes which know about the colour filter'''
    
    def Generate(self, event=None):
        dlg = genImageDialog.GenImageDialog(self.visFr, mode=self.mode)
        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()

            bCurr = wx.BusyCursor()

            im = self.genIm(dlg)

            x0 = max(self.visFr.glCanvas.xmin, self.visFr.imageBounds.x0)
            y0 = max(self.visFr.glCanvas.ymin, self.visFr.imageBounds.y0)
            x1 = min(self.visFr.glCanvas.xmax, self.visFr.imageBounds.x1)
            y1 = min(self.visFr.glCanvas.ymax, self.visFr.imageBounds.y1)

            #imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)
            imb = ImageBounds(x0, y0, x1, y1)

            img = GeneratedImage(im,imb, pixelSize )
            imf = imageView.ImageViewFrame(self,img, self.visFr.glCanvas)
            #self.generatedImages.append(imf)
            imf.Show()

            self.visFr.RefreshView()

        dlg.Destroy()
        return imf
    
    def Generate(self, event=None):
        dlg = genImageDialog.GenImageDialog(self.visFr, mode=self.mode, colours=self.visFr.fluorSpecies.keys())
        ret = dlg.ShowModal()

        bCurr = wx.BusyCursor()

        if ret == wx.ID_OK:
            pixelSize = dlg.getPixelSize()

            status = statusLog.StatusLogger('Generating %s Image ...' % self.name)

            x0 = max(self.glCanvas.xmin, self.imageBounds.x0)
            y0 = max(self.glCanvas.ymin, self.imageBounds.y0)
            x1 = min(self.glCanvas.xmax, self.imageBounds.x1)
            y1 = min(self.glCanvas.ymax, self.imageBounds.y1)

            #imb = ImageBounds(self.glCanvas.xmin,self.glCanvas.ymin,self.glCanvas.xmax,self.glCanvas.ymax)
            imb = ImageBounds(x0, y0, x1, y1)

            colours =  dlg.getColour()
            oldC = self.colourFilter.currentColour

            ims = []

            for c in  colours:
                self.colourFilter.setColour(c)
                #im = visHelpers.rendHist(self.colourFilter['x'],self.colourFilter['y'], imb, pixelSize)
                im = self.genIm(dlg)

                ims.append(GeneratedImage(im,imb, pixelSize ))

            imfc = imageView.MultiChannelImageViewFrame(self.visFr, self.visFr.glCanvas, ims, colours, title='Generated %s - %3.1fnm bins' % (self.name, pixelSize))

            #self.generatedImages.append(imfc)
            imfc.Show()

            self.colourFilter.setColour(oldC)

        dlg.Destroy()
        return imfc


