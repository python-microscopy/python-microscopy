#!/usr/bin/python

##################
# imageView.py
#
# Copyright David Baddeley, 2009
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

#import math
import numpy
import os
#import sys
import wx
#import wx.lib.agw.aui as aui
#import histLimits
# import pylab
import scipy.misc
#import subprocess

from PYME.ui import wx_compat
from PYME.ui import selection

#from PYME.Analysis import thresholding

#from PYME.DSView.myviewpanel_numarray import MyViewPanel
#from PYME.DSView.arrayViewPanel import ArrayViewPanel
from PYME.DSView.displayOptions import DisplayOpts
from PYME.DSView.DisplayOptionsPanel import OptionsPanel
#from PYME.IO.image import ImageStack

#from PYME.misc.auiFloatBook import AuiNotebookWithFloatingPages

class ImageViewPanel(wx.Panel):
    def __init__(self, parent, image, glCanvas, do, chan=0, zdim=2):
        wx.Panel.__init__(self, parent, -1, size=parent.Size)

        self.image = image
        self.glCanvas = glCanvas

        self.do = do
        self.chan = chan
        self.zdim = zdim
        
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)

        wx.EVT_LEFT_DOWN(self, self.OnLeftDown)
        wx.EVT_LEFT_UP(self, self.OnLeftUp)
        wx.EVT_MIDDLE_DOWN(self, self.OnMiddleDown)
        wx.EVT_MIDDLE_UP(self, self.OnMiddleUp)
        wx.EVT_MOTION(self, self.OnMotion)

        self.selecting=False
        self.panning = False

        self.do.WantChangeNotification.append(self.Refresh)


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

        step = int(step)
        x0_p = int(x0_ / self.image.pixelSize)
        x1_p = int(x1_ / self.image.pixelSize)
        y0_p = int(y0_ / self.image.pixelSize)
        y1_p = int(y1_ / self.image.pixelSize)
        
        #if self.image.data.shape) == 2:
        #    im = numpy.flipud(self.image.img[int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step].astype('f').T)
        if self.zdim ==0:
            im = numpy.flipud(self.image.data[self.do.zp,x0_p:x1_p:step, y0_p:y1_p:step, self.chan].squeeze().astype('f').T)
        else:
            im = numpy.flipud(self.image.data[x0_p:x1_p:step, y0_p:y1_p:step, self.do.zp, self.chan].squeeze().astype('f').T)

        im = im - self.do.Offs[self.chan] #self.clim[0]
        im = im*self.do.Gains[self.chan]    #/(self.clim[1] - self.clim[0])

        #print((im.shape))

        im = (255*self.do.cmaps[self.chan](im)[:,:,:3]).astype('b')
            
        imw =  wx_compat.ImageFromData(im.shape[1], im.shape[0], im.ravel())
                                         
        imw.Rescale(imw.GetWidth()*sc,imw.GetHeight()*sc)
        self.curIm = imw

        dc.Clear()
        
        dc.DrawBitmap(wx_compat.BitmapFromImage(imw),(-self.centreX + x0 + width/2)/pixelsize,(self.centreY - y1 + height/2)/pixelsize)

    def DrawOverlays(self,dc):
        sc = self.image.pixelSize/self.glCanvas.pixelsize
        
        if self.glCanvas.centreCross:
            print('drawing crosshair')
            dc.SetPen(wx.Pen(wx.GREEN, 2))

            dc.DrawLine(.5*self.Size[0], 0, .5*self.Size[0], self.Size[1])
            dc.DrawLine(0, .5*self.Size[1], self.Size[0], .5*self.Size[1])

            dc.SetPen(wx.NullPen)

        if self.do.showSelection:
            col = wx.TheColourDatabase.FindColour('YELLOW')
            #col.Set(col.red, col.green, col.blue, 125)
            dc.SetPen(wx.Pen(col,1))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            lx, ly, hx, hy = self.do.GetSliceSelection()

            lx, ly = self._PixelToScreenCoordinates(lx, ly)
            hx, hy = self._PixelToScreenCoordinates(hx, hy)


            if self.do.selection.mode == selection.SELECTION_RECTANGLE:
                dc.DrawRectangle(lx,ly, (hx-lx),(hy-ly))
                
            elif self.do.selection.mode == selection.SELECTION_SQUIGGLE:
                if len(self.do.selection.trace) > 2:
                    x, y = numpy.array(self.do.selection.trace).T
                    pts = numpy.vstack(self._PixelToScreenCoordinates(x, y)).T
                    print((pts.shape))
                    dc.DrawSpline(pts)
            elif self.do.selection.width == 1:
                dc.DrawLine(lx,ly, hx,hy)
            else:
                dx = hx - lx
                dy = hy - ly

                w = self.do.selection.width*sc

                if dx == 0 and dy == 0: #special case - profile is orthogonal to current plane
                    d_x = 0.5*w
                    d_y = 0.5*w
                else:
                    d_x = 0.5*w*dy/numpy.sqrt((dx**2 + dy**2))
                    d_y = 0.5*w*dx/numpy.sqrt((dx**2 + dy**2))

                
                x_1 = lx
                y_1 = ly
                x_2 = hx
                y_2 = hy

                dc.DrawLine(lx,ly, hx,hy)
                dc.DrawPolygon([(x_1 +d_x, y_1-d_y), (x_1 - d_x, y_1 + d_y), (x_2-d_x, y_2+d_y), (x_2 + d_x, y_2 - d_y)])


            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)

            
        
    def OnPaint(self,event):
        DC = wx.PaintDC(self)
        #self.PrepareDC(DC)
        
        s = self.GetVirtualSize()
        MemBitmap = wx_compat.EmptyBitmap(s.GetWidth(), s.GetHeight())
        #del DC
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)
        try:
            #DC.BeginDrawing()
            
            self.DoPaint(MemDC)
            self.DrawOverlays(MemDC)
            
            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            #DC.EndDrawing()
        finally:
            
            del MemDC
            del MemBitmap

    def OnSize(self, event):
        self.Refresh()
        self.Update()

    def _ScreenToAbsCoordinates(self, x, y):
        xp = self.centreX + self.glCanvas.pixelsize*(x - self.Size[0]/2)
        yp = self.centreY - self.glCanvas.pixelsize*(y - self.Size[1]/2)

        return xp, yp

    def _ScreenToPixelCoordinates(self, x, y):
        xp, yp = self._ScreenToAbsCoordinates(x, y)
        return (xp - self.image.imgBounds.x0)/self.image.pixelSize , (yp- self.image.imgBounds.y0)/self.image.pixelSize

    def _AbsToScreenCoordinates(self, x, y):
        xp = (x -self.centreX)/self.glCanvas.pixelsize + self.Size[0]/2
        yp = (-y + self.centreY)/self.glCanvas.pixelsize + self.Size[1]/2

        return xp, yp

    def _PixelToScreenCoordinates(self, x, y):
        return self._AbsToScreenCoordinates(x*self.image.pixelSize + self.image.imgBounds.x0, y*self.image.pixelSize + self.image.imgBounds.y0)

    def OnWheel(self, event):
        rot = event.GetWheelRotation()

        #get translated coordinates
        #xp = self.centreX + self.glCanvas.pixelsize*(event.GetX() - self.Size[0]/2)
        #yp = self.centreY - self.glCanvas.pixelsize*(event.GetY() - self.Size[1]/2)

        xp, yp = self._ScreenToAbsCoordinates(event.GetX(), event.GetY())

        #print xp
        #print yp
        self.glCanvas.WheelZoom(rot, xp, yp)
        
        

    def OnKeyPress(self, event):
        if event.GetKeyCode() == wx.WXK_PAGEUP:
            self.do.zp =max(self.do.zp - 1, 0)
            self.Refresh()
            self.Update()
        elif event.GetKeyCode() == wx.WXK_PAGEDOWN:
            self.do.zp = min(self.do.zp + 1, self.image.data.shape[self.zdim] - 1)
            self.Refresh()
            self.Update()
        else:
            event.Skip()

    def OnLeftDown(self,event):
        if self.do.leftButtonAction == self.do.ACTION_SELECTION:
            self.StartSelection(event)
            
        event.Skip()

    def OnLeftUp(self,event):
        if self.do.leftButtonAction == self.do.ACTION_SELECTION:
            self.ProgressSelection(event)
            self.EndSelection()
        else:
            pass
            #self.OnSetPosition(event)
        event.Skip()

    def OnMiddleDown(self, event):
        x, y = self._ScreenToAbsCoordinates(event.GetX(), event.GetY())
        self.xDragStart = x
        self.yDragStart = y

        self.panning = True
        event.Skip()

    def OnMiddleUp(self, event):
        self.panning = False
        event.Skip()

    def StartSelection(self,event):
        self.selecting = True

        xp, yp = self._ScreenToPixelCoordinates(event.GetX(), event.GetY())
        
        self.do.selection.start.x = int(xp)
        self.do.selection.start.y = int(yp)
        
        self.do.selection.trace = []
        self.do.selection.trace.append((xp, yp))

    def OnMotion(self, event):
        if event.Dragging() and self.selecting:
            self.ProgressSelection(event)

        elif event.Dragging() and self.panning:
            x, y = self._ScreenToAbsCoordinates(event.GetX(), event.GetY())
            #x = event.GetX()
            #y = event.GetY()

            dx = (x - self.xDragStart)
            dy = (y - self.yDragStart)

            self.xDragStart = x
            self.yDragStart = y

            self.glCanvas.pan(-dx, -dy)

    def ProgressSelection(self,event):
        xp, yp = self._ScreenToPixelCoordinates(event.GetX(), event.GetY())

        if not event.ShiftDown():
            self.do.selection.finish.x = int(xp)
            self.do.selection.finish.y = int(yp)
            
        else: #lock
            self.do.selection.finish.x = int(xp)
            self.do.selection.finish.y = int(yp)

            dx = abs(self.do.selection.finish.x - self.do.selection.start.x)
            dy = abs(self.do.selection.finish.y - self.do.selection.start.y)

            if dx > 1.5*dy: #horizontal
                self.do.selection.finish.y = self.do.selection.start.y
            elif dy > 1.5*dx: #vertical
                self.do.selection.finish.x = self.do.selection.start.x
            else: #diagonal
                self.do.selection.finish.y = self.do.selection.start.y + dx*numpy.sign(self.do.selection.finish.y - self.do.selection.start.y)
                
        self.do.selection.trace.append((xp, yp))

        self.Refresh()
        self.Update()

    def EndSelection(self):
        self.selecting = False

class ColourImageViewPanel(ImageViewPanel):
    def __init__(self, parent, glCanvas, do, image, zdim=2):
        ImageViewPanel.__init__(self, parent, image, glCanvas, do, chan=0, zdim=zdim)

    def DoPaint(self, dc):
        pixelsize = self.glCanvas.pixelsize

        self.centreX = (self.glCanvas.xmin + self.glCanvas.xmax)/2.
        self.centreY = (self.glCanvas.ymin + self.glCanvas.ymax)/2.
        #print centreX

        width = self.Size[0]*pixelsize
        height = self.Size[1]*pixelsize

        im_ = numpy.zeros((self.Size[0], self.Size[1], 3), 'uint8')

        #for ivp in self.ivps:
        for chanNum in range(self.image.data.shape[3]):
            #img = ivp.image
            #clim = ivp.clim
            cmap = self.do.cmaps[chanNum]

            x0 = max(self.centreX  - width/2, self.image.imgBounds.x0)
            x1 = min(self.centreX  + width/2, self.image.imgBounds.x1)
            y0 = max(self.centreY  - height/2, self.image.imgBounds.y0)
            y1 = min(self.centreY  + height/2, self.image.imgBounds.y1)

            x0_ = x0 - self.image.imgBounds.x0
            x1_ = x1 - self.image.imgBounds.x0
            y0_ = y0 - self.image.imgBounds.y0
            y1_ = y1 - self.image.imgBounds.y0

            sc = float(self.image.pixelSize/pixelsize)

            if sc >= 1:
                step = 1
            else:
                step = 2**(-numpy.ceil(numpy.log2(sc)))
                sc = sc*step

            step = int(step)

            #if len(img.img.shape) == 2:
            #    im = numpy.flipud(img.img[int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step].astype('f').T)
            if self.zdim ==0:
                im = numpy.flipud(self.image.data[self.do.zp,int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step, chanNum].squeeze().astype('f').T)
            else:
                im = numpy.flipud(self.image.data[int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step, self.do.zp, chanNum].squeeze().astype('f').T)

            #print clim
            #im = im.astype('f') - clim[0]
            #im = im/(clim[1] - clim[0])
            im = im - self.do.Offs[chanNum] #self.clim[0]
            im = im*self.do.Gains[chanNum]    #/(self.clim[1] - self.clim[0])

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

            dx = int(round((-self.centreX + x0 + width/2)/pixelsize))
            dy = int(round((self.centreY - y1 + height/2)/pixelsize))

            #print dx, dy, im_.shape, im_[dx:(im.shape[0] + dx), dy:(im.shape[1] + dy), :].shape
            #print self.centreX, self.centreY, x0, y0, dx, dy, width, height, pixelsize, round((-self.centreX + x0 + width/2)/pixelsize)

            im_[dy:(im.shape[0] + dy), dx:(im.shape[1] + dx), :] = im_[dy:(im.shape[0] + dy), dx:(im.shape[1] + dx), :] + im[:(im_.shape[0] - dy),:(im_.shape[1] - dx)]

            #print im.shape

        im_ = numpy.minimum(im_, 255).astype('b')
        #print im_.shape

        imw =  wx_compat.ImageFromData(im_.shape[1], im_.shape[0], im_.ravel())
        #print imw.GetWidth()

        #imw.Rescale(imw.GetWidth()*sc,imw.GetHeight()*sc)
            #print imw.Size
        self.curIm = imw

        dc.Clear()

        #dc.DrawBitmap(wx_compat.BitmapFromImage(imw),(-self.centreX + x0 + width/2)/pixelsize,(self.centreY - y1 + height/2)/pixelsize)
        dc.DrawBitmap(wx_compat.BitmapFromImage(imw), 0,0)

        #print self.glCanvas.centreCross

        if self.glCanvas.centreCross:
            print('drawing crosshair')
            dc.SetPen(wx.Pen(wx.GREEN, 2))

            dc.DrawLine(.5*self.Size[0], 0, .5*self.Size[0], self.Size[1])
            dc.DrawLine(0, .5*self.Size[1], self.Size[0], .5*self.Size[1])

            dc.SetPen(wx.NullPen)
        
        
#from PYME.IO.MetaDataHandler import NestedClassMDHandler
#from PYME.DSView import modules as dsvmods
#from PYME.DSView import ViewIm3D

#def MultiChannelImageViewFrame(parent, glCanvas, image, title='Generated Image',zp=0, zdim=2):
#    return ViewIm3D(image, mode='visGUI', title=title, glCanvas=glCanvas, parent=parent)

#class MultiChannelImageViewFrame(wx.Frame):
#    def __init__(self, parent, glCanvas, image, title='Generated Image',zp=0, zdim=2):
#        wx.Frame.__init__(self, parent, -1, title=title, size=(800,800))
#
#        self.glCanvas = glCanvas
#        self.parent = parent
#        self.frame = self
#
#        self.image = image
#
#        self.glCanvas.wantViewChangeNotification[self.image.filename] = self
#
##        self.image = ImageStack([numpy.atleast_3d(im.img) for im in images])
##        self.image.pixelSize = images[0].pixelSize
##        self.image.sliceSize = images[0].sliceSize
##
##        self.image.imgBounds = images[0].imgBounds
##
##        #self.images = images
##        self.image.names = [n or 'Image' for n in names]
#
#        #md = NestedClassMDHandler()
##        self.image.mdh.setEntry('voxelsize.x', .001*images[0].pixelSize)
##        self.image.mdh.setEntry('voxelsize.y', .001*images[0].pixelSize)
##        self.image.mdh.setEntry('voxelsize.z', .001*images[0].sliceSize)
##
##        self.image.mdh.setEntry('ChannelNames', self.image.names)
#
#        #self.image = ImageStack([numpy.atleast_3d(im.img) for im in images], md)
#
#        self.paneHooks = []
#        self.updateHooks = []
#        self.statusHooks = []
#        self.installedModules = []
#
#        #self.notebook = AuiNotebookWithFloatingPages(id=-1, parent=self, style=wx.aui.AUI_NB_TAB_SPLIT)
#        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT | aui.AUI_MGR_AUTONB_NO_CAPTION)
#        atabstyle = self._mgr.GetAutoNotebookStyle()
#        self._mgr.SetAutoNotebookStyle((atabstyle ^ aui.AUI_NB_BOTTOM) | aui.AUI_NB_TOP)
#        # tell AuiManager to manage this frame
#        self._mgr.SetManagedWindow(self)
#
#        self.ivps = []
#        self.pane0 = None
#
#        asp = self.image.sliceSize/self.image.pixelSize
#        if asp == 0:
#            asp = 1
#        #ims = self.images[0].img.shape
#
#        #if len(ims) == 2:
#        #    ims = list(ims) + [1]
#
#        self.do = DisplayOpts(self.image.data, asp)
#        self.do.Optimise()
#        self.do.zp = zp
#        self.do.names = self.image.names
#
##        cmaps = [pylab.cm.r, pylab.cm.g, pylab.cm.b]
##
##        for name, i in zip(self.image.names, xrange(self.image.data.shape[3])):
##            self.ivps.append(ImageViewPanel(self, self.image, glCanvas, self.do, chan=i, zdim=zdim))
##            if self.image.data.shape[3] > 1 and len(cmaps) > 0:
##                self.do.cmaps[i] = cmaps.pop(0)
##
##            self.AddPage(page=self.ivps[-1], select=True, caption=name)
##
##
##        if self.image.data.shape[2] > 1:
##            self.AddPage(page=ArrayViewPanel(self, do=self.do, aspect = asp), select=False, caption='Slices')
##
##        elif self.image.data.shape[3] > 1:
##            self.civp = ColourImageViewPanel(self, glCanvas, self.do, self.image, zdim=zdim)
##            self.civp.ivps = self.ivps
##            self.AddPage(page=self.civp, select=True, caption='Composite')
#
#
#
#        self.menubar = self.CreateMenuBar()
#        self.SetMenuBar(self.menubar)
#
#        self.limitsFrame = None
#
#        self.optionspanel = OptionsPanel(self, self.do, thresholdControls=True)
#        self.optionspanel.SetSize(self.optionspanel.GetBestSize())
#        pinfo = aui.AuiPaneInfo().Name("optionsPanel").Right().Caption('Display Settings').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
#        self._mgr.AddPane(self.optionspanel, pinfo)
#
#        self._mgr.AddPane(self.optionspanel.CreateToolBar(self), aui.AuiPaneInfo().Name("ViewTools").Caption("View Tools").CloseButton(False).
#                      ToolbarPane().Right().GripperTop())
#
#        if self.do.ds.shape[2] > 1:
#            from PYME.DSView.modules import playback
#            self.playbackpanel = playback.PlayPanel(self, self)
#            self.playbackpanel.SetSize(self.playbackpanel.GetBestSize())
#
#            pinfo1 = aui.AuiPaneInfo().Name("playbackPanel").Bottom().Caption('Playback').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
#            self._mgr.AddPane(self.playbackpanel, pinfo1)
#            self.do.WantChangeNotification.append(self.playbackpanel.update)
#
#        self.mode = 'visGUI'
#        dsvmods.loadMode('visGUI', self)
#        self.CreateModuleMenu()
#
#        self._mgr.Update()
#        self._mgr.MinimizePane(pinfo)
#
#        wx.EVT_CLOSE(self, self.OnClose)
#        self.do.WantChangeNotification.append(self.Refresh)
#
#    def CreateModuleMenu(self):
#        self.modMenuIds = {}
#        self.mModules = wx.Menu()
#        for mn in dsvmods.allmodules:
#            id = wx.NewId()
#            self.mModules.AppendCheckItem(id, mn)
#            self.modMenuIds[id] = mn
#            if mn in self.installedModules:
#                self.mModules.Check(id, True)
#
#            wx.EVT_MENU(self, id, self.OnToggleModule)
#
#        self.menubar.Append(self.mModules, "&Modules")
#
#    def OnToggleModule(self, event):
#        id = event.GetId()
#        mn = self.modMenuIds[id]
#        if self.mModules.IsChecked(id):
#            dsvmods.loadModule(mn, self)
#
#        if mn in self.installedModules:
#            self.mModules.Check(id, True)
#
#        #self.CreateFoldPanel()
#        self._mgr.Update()
#
#
#
#    def AddPage(self, page=None, select=True,caption='Dummy'):
#        if self.pane0 == None:
#            name = caption.replace(' ', '')
#            self._mgr.AddPane(page, aui.AuiPaneInfo().
#                          Name(name).Caption(caption).CaptionVisible(False).Centre().CloseButton(False))
#            self.pane0 = name
#
#        else:
#            self._mgr.Update()
#            pn = self._mgr.GetPaneByName(self.pane0)
#            if pn.IsNotebookPage():
#                print pn.notebook_id
#                nbs = self._mgr.GetNotebooks()
#                if len(nbs) > pn.notebook_id:
#                    currPage = nbs[pn.notebook_id].GetSelection()
#                self._mgr.AddPane(page, aui.AuiPaneInfo().
#                              Name(caption.replace(' ', '')).Caption(caption).CloseButton(False).NotebookPage(pn.notebook_id))
#                if (not select) and len(nbs) > pn.notebook_id:
#                    nbs[pn.notebook_id].SetSelection(currPage)
#            else:
#                self._mgr.AddPane(page, aui.AuiPaneInfo().
#                              Name(caption.replace(' ', '')).Caption(caption).CloseButton(False), target=pn)
#                #nb = self._mgr.GetNotebooks()[0]
#                #if not select:
#                #    nb.SetSelection(0)
#
#        self._mgr.Update()
#
#    def update(self):
#        #if 'playbackpanel' in dir(self):
#        #        self.playbackpanel.update()
#        self.Refresh()
#
#    def GetSelectedPage(self):
#        nbs = self._mgr.GetNotebooks()
#        currPage = nbs[0].GetCurrentPage()
#
#        return currPage
#
#
#    def CreateMenuBar(self):
#
#        # Make a menubar
#        file_menu = wx.Menu()
#
#        #ID_SAVE = wx.NewId()
#        #ID_CLOSE = wx.NewId()
#        ID_EXPORT = wx.NewId()
#        #ID_SAVEALL = wx.NewId()
#
#        #ID_VIEW_CONSOLE = wx.NewId()
#        ID_VIEW_BACKGROUND = wx.NewId()
#        ID_FILTER_GAUSS = wx.NewId()
#
#        #self.ID_VIEW_CMAP_INVERT = wx.NewId()
#
#        file_menu.Append(wx.ID_SAVE, "&Save")
#        #file_menu.Append(ID_SAVEALL, "Save &Multi-channel")
#        file_menu.Append(ID_EXPORT, "E&xport Current View")
#
#        file_menu.AppendSeparator()
#
#        file_menu.Append(wx.ID_CLOSE, "&Close")
#
#        view_menu = wx.Menu()
#        #view_menu.Append(ID_VIEW_CONSOLE, "&Console")
#        view_menu.Append(ID_VIEW_BACKGROUND, "Set as visualisation &background")
#
#        self.mProcessing = wx.Menu()
#        #self.mProcessing.Append(ID_FILTER_GAUSS, "&Gaussian Filter")
#
#        menu_bar = wx.MenuBar()
#
#        menu_bar.Append(file_menu, "&File")
#        menu_bar.Append(view_menu, "&View")
#        menu_bar.Append(self.mProcessing, "&Processing")
#        #menu_bar.Append(td_menu, "&3D")
#
#        self.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
#        #self.Bind(wx.EVT_MENU, self.OnSaveChannels, id=ID_SAVEALL)
#        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
#        self.Bind(wx.EVT_MENU, self.OnExport, id=ID_EXPORT)
#        #self.Bind(wx.EVT_MENU, self.OnViewCLim, id=ID_VIEW_COLOURLIM)
#        #self.Bind(wx.EVT_MENU, self.OnViewConsole, id=ID_VIEW_CONSOLE)
#        self.Bind(wx.EVT_MENU, self.OnViewBackground, id=ID_VIEW_BACKGROUND)
#        #self.Bind(wx.EVT_MENU, self.OnGaussianFilter, id=ID_FILTER_GAUSS)
#
#        return menu_bar
#
#
#    def OnSave(self, event):
#        self.image.Save()
#        self.SetTitle(self.image.filename)
#
#    def OnViewBackground(self, event):
#        ivp = self.GetSelectedPage() #self.notebook.GetPage(self.notebook.GetSelection())
#
#        if 'image' in dir(ivp): #is a single channel
#            img = numpy.minimum(255.*(ivp.image.img - ivp.clim[0])/(ivp.clim[1] - ivp.clim[0]), 255).astype('uint8')
#            self.glCanvas.setBackgroundImage(img, (ivp.image.imgBounds.x0, ivp.image.imgBounds.y0), pixelSize=ivp.image.pixelSize)
#
#        self.glCanvas.Refresh()
#
#
#    def OnClose(self, event):
#        #if self in self.parent.generatedImages:
#        #    self.parent.generatedImages.remove(self)
#        self.Destroy()
#
#    def OnExport(self, event):
#        ivp = self.notebook.GetPage(self.notebook.GetSelection())
#        fname = wx.FileSelector('Save Current View', default_extension='.tif', wildcard="Supported Image Files (*.tif, *.bmp, *.gif, *.jpg, *.png)|*.tif, *.bmp, *.gif, *.jpg, *.png", flags = wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
#
#        if not fname == "":
#            ext = os.path.splitext(fname)[-1]
#            if ext == '.tif':
#                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_TIF)
#            elif ext == '.png':
#                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_PNG)
#            elif ext == '.jpg':
#                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_JPG)
#            elif ext == '.gif':
#                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_GIF)
#            elif ext == '.bmp':
#                ivp.curIm.SaveFile(fname, wx.BITMAP_TYPE_BMP)
#
#
##    def OnGaussianFilter(self, event):
##        from scipy.ndimage import gaussian_filter
##        from PYME.LMVis.visHelpers import ImageBounds, GeneratedImage
##
##        dlg = wx.TextEntryDialog(self, 'Blur size [pixels]:', 'Gaussian Blur', '[1,1,1]')
##
##        if dlg.ShowModal() == wx.ID_OK:
##            sigmas = eval(dlg.GetValue())
##            #print sigmas
##            #print self.images[0].img.shape
##            filt_ims = [GeneratedImage(gaussian_filter(self.image.data[:,:,:,chanNum].squeeze(), sigmas), self.image.imgBounds, self.image.pixelSize, self.image.sliceSize) for chanNum in range(self.image.data.shape[3])]
##
##            imfc = MultiChannelImageViewFrame(self.parent, self.parent.glCanvas, filt_ims, self.image.names, title='Filtered Image - %3.1fnm bins' % self.image.pixelSize)
##
##            self.parent.generatedImages.append(imfc)
##            imfc.Show()
##
##        dlg.Destroy()
