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

import numpy

import wx
from PYME.ui import wx_compat
from PYME.ui import selection

# import scipy.misc
import scipy.ndimage

from PYME.DSView.displayOptions import DisplayOpts


class ImageViewPanel(wx.Panel):
    def __init__(self, parent, image, glCanvas, do, chan=0, zdim=2):
        wx.Panel.__init__(self, parent, -1, size=parent.Size)

        self.image = image
        self.glCanvas = glCanvas

        self.do = do
        self.chan = chan
        self.zdim = zdim
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyPress)

        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.OnMiddleDown)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnMiddleUp)
        self.Bind(wx.EVT_MOTION, self.OnMotion)

        self.selecting=False
        self.panning = False

        self.do.WantChangeNotification.append(self.Refresh)
    
    def DrawOverlays(self,dc):
        sc = self.image.pixelSize/self.glCanvas.pixelsize
        
        if False:#self.glCanvas.centreCross:
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
                dc.DrawRectangle(int(lx),int(ly), int(hx-lx),int(hy-ly))
                
            elif self.do.selection.mode == selection.SELECTION_SQUIGGLE:
                if len(self.do.selection.trace) > 2:
                    x, y = numpy.array(self.do.selection.trace).T
                    pts = numpy.vstack(self._PixelToScreenCoordinates(x, y)).T
                    # print((pts.shape))
                    dc.DrawSpline(pts.astype('i'))
            elif self.do.selection.width == 1:
                dc.DrawLine(int(lx),int(ly),int(hx),int(hy))
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

                dc.DrawLine(int(lx),int(ly), int(hx),int(hy))
                dc.DrawPolygon([(int(x_1 +d_x), int(y_1-d_y)), (int(x_1 - d_x), int(y_1 + d_y)), (int(x_2-d_x), int(y_2+d_y)), (int(x_2 + d_x), int(y_2 - d_y))])


            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)

            
        
    def OnPaint(self,event):
        if not self.IsShownOnScreen():
            return
        
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
        yp = self.centreY + self.glCanvas.pixelsize*(y - self.Size[1]/2)

        return xp, yp

    def _ScreenToPixelCoordinates(self, x, y):
        xp, yp = self._ScreenToAbsCoordinates(x, y)
        return (xp - self.image.imgBounds.x0)/self.image.pixelSize , (yp- self.image.imgBounds.y0)/self.image.pixelSize

    def _AbsToScreenCoordinates(self, x, y):
        xp = (x -self.centreX)/self.glCanvas.pixelsize + self.Size[0]/2
        yp = (y - self.centreY)/self.glCanvas.pixelsize + self.Size[1]/2

        return xp, yp

    def _PixelToScreenCoordinates(self, x, y):
        return self._AbsToScreenCoordinates(x*self.image.pixelSize + self.image.imgBounds.x0, y*self.image.pixelSize + self.image.imgBounds.y0)

    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        xp, yp = self._ScreenToAbsCoordinates(event.GetX(), event.GetY())

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
        
    def _map_image(self, im, chan):
        im = numpy.clip((im - self.do.Offs[chan])*self.do.Gains[chan], 0, 1)
    
        return (255 * self.do.cmaps[chan](im)[:, :, :3]).astype('b')
    
    def _coordinates(self):
        pixelsize = float(self.glCanvas.pixelsize)
        centre_x = (self.glCanvas.xmin + self.glCanvas.xmax) / 2.
        centre_y = (self.glCanvas.ymin + self.glCanvas.ymax) / 2.

        width = self.Size[0] * pixelsize
        height = self.Size[1] * pixelsize
        
        bbox_nm = (max(centre_x - width / 2, self.image.imgBounds.x0),
                   min(centre_x + width / 2, self.image.imgBounds.x1),
                   max(centre_y - height / 2, self.image.imgBounds.y0),
                   min(centre_y + height / 2, self.image.imgBounds.y1))

        sc = float(self.image.pixelSize / pixelsize)

        if sc >= 1:
            step = 1
        else:
            step = 2 ** (-numpy.ceil(numpy.log2(sc)))
            sc = sc * step

        step = int(step)
        
        return pixelsize, (centre_x, centre_y), (width, height), bbox_nm, sc, step

    def DoPaint(self, dc):
        pixelsize, (self.centreX, self.centreY), (width, height), ( x0, x1, y0, y1), sc, step = self._coordinates()
        
        x0_ = x0 - self.image.imgBounds.x0
        x1_ = x1 - self.image.imgBounds.x0
        y0_ = y0 - self.image.imgBounds.y0
        y1_ = y1 - self.image.imgBounds.y0

        x0_p = int(x0_ / self.image.pixelSize)
        x1_p = int(x1_ / self.image.pixelSize)
        y0_p = int(y0_ / self.image.pixelSize)
        y1_p = int(y1_ / self.image.pixelSize)
    
        if self.zdim == 0:
            im = (self.image.data[self.do.zp, x0_p:x1_p:step, y0_p:y1_p:step, self.chan].squeeze().astype('f').T)
        else:
            im = (self.image.data[x0_p:x1_p:step, y0_p:y1_p:step, self.do.zp, self.chan].squeeze().astype('f').T)
    
        im = self._map_image(im, self.chan)
    
        imw = wx_compat.ImageFromData(im.shape[1], im.shape[0], im.ravel())
        imw.Rescale(int(imw.GetWidth() * sc), int(imw.GetHeight() * sc))
        self.curIm = imw
    
        dc.Clear()
        dc.DrawBitmap(wx_compat.BitmapFromImage(imw), int((-self.centreX + x0 + width / 2) / pixelsize),
                      int((-self.centreY + y0 + height / 2) / pixelsize))


class ColourImageViewPanel(ImageViewPanel):
    def __init__(self, parent, glCanvas, do, image, zdim=2):
        ImageViewPanel.__init__(self, parent, image, glCanvas, do, chan=0, zdim=zdim)

    def DoPaint(self, dc):
        pixelsize, (self.centreX, self.centreY), (width, height), (x0, x1, y0, y1), sc, step = self._coordinates()

        x0_ = x0 - self.image.imgBounds.x0
        x1_ = x1 - self.image.imgBounds.x0
        y0_ = y0 - self.image.imgBounds.y0
        y1_ = y1 - self.image.imgBounds.y0

        im_ = numpy.zeros((self.Size[1], self.Size[0], 3), 'uint8')

        for chanNum in range(self.image.data.shape[3]):
            if self.zdim ==0:
                im = (self.image.data[self.do.zp,int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step, chanNum].squeeze().astype('f').T)
            else:
                im = (self.image.data[int(x0_ / self.image.pixelSize):int(x1_ / self.image.pixelSize):step, int(y0_ / self.image.pixelSize):int(y1_ / self.image.pixelSize):step, self.do.zp, chanNum].squeeze().astype('f').T)

            # im = self._map_image(scipy.misc.imresize(im, sc), chanNum)
            im = self._map_image(scipy.ndimage.zoom(im, sc), chanNum)
        
            dx = int(round((-self.centreX + x0 + width/2)/pixelsize))
            dy = int(round((self.centreY - y1 + height/2)/pixelsize))

            im_[dy:(im.shape[0] + dy), dx:(im.shape[1] + dx), :] = im_[dy:(im.shape[0] + dy), dx:(im.shape[1] + dx), :] + im[:(im_.shape[0] - dy),:(im_.shape[1] - dx)]


        im_ = numpy.minimum(im_, 255).astype('b')

        imw =  wx_compat.ImageFromData(im_.shape[1], im_.shape[0], im_.ravel())
        self.curIm = imw

        dc.Clear()
        dc.DrawBitmap(wx_compat.BitmapFromImage(imw), 0,0)
        
        
        
