#!/usr/bin/python

##################
# myviewpanel_numarray.py
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

import wx
import wx.lib.agw.aui as aui

#import sys
import os
#sys.path.append(".")
from PYME.DSView import scrolledImagePanel
from PYME.DSView.displayOptions import DisplayOpts, labeled
from PYME.DSView.DisplayOptionsPanel import OptionsPanel
from PYME.DSView.OverlaysPanel import OverlayPanel

from PYME.DSView.modules import playback
from PYME.DSView.LUT import applyLUT

import numpy
import scipy
# import pylab
import matplotlib.pyplot as plt
import matplotlib.cm

from PYME.contrib import dispatch

LUTCache = {}

SLICE_AXIS_LUT = {DisplayOpts.SLICE_XY:2, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:0}
TOL_AXIS_LUT = {DisplayOpts.SLICE_XY:0, DisplayOpts.SLICE_XZ:1,DisplayOpts.SLICE_YZ:2}

def getLUT(cmap):
    if not cmap.name in LUTCache.keys():
        #calculate and cache LUT
        LUTCache[cmap.name] = (255*(cmap(numpy.linspace(0,1,256))[:,:3].T)).copy().astype('uint8')

    return LUTCache[cmap.name]



            
class ArrayViewPanel(scrolledImagePanel.ScrolledImagePanel):
    def __init__(self, parent, dstack = None, aspect=1, do = None, voxelsize=[1,1,1]):
        
        if (dstack is None and do is None):
            dstack = scipy.zeros((10,10))

        if do is None:
            self.do = DisplayOpts(dstack, aspect=aspect)
            self.do.Optimise()
        else:
            self.do = do
            
        self.voxelsize = voxelsize

        scrolledImagePanel.ScrolledImagePanel.__init__(self, parent, self.DoPaint, style=wx.SUNKEN_BORDER|wx.TAB_TRAVERSAL)

        self.do.WantChangeNotification.append(self.GetOpts)
        #self.do.WantChangeNotification.append(self.Refresh)

        self.SetVirtualSize(wx.Size(self.do.ds.shape[0],self.do.ds.shape[1]))
        #self.imagepanel.SetSize(wx.Size(self.do.ds.shape[0],self.do.ds.shape[1]))
        

        self.points =[]
        self.pointsR = []
        self.showPoints = True
        self.showTracks = True
        self.showContours = True
        self.showScaleBar = True
        self.scaleBarLength = 2000
        self.pointMode = 'confoc'
        self.pointTolNFoc = {'confoc' : (5,5,5), 'lm' : (2, 5, 5), 'splitter' : (2,5,5)}
        self.showAdjacentPoints = False
        self.pointSize = 11
        self.layerMode = 'Add'

        self.psfROIs = []
        self.psfROISize=[30,30,30]

        self.lastUpdateTime = 0
        self.lastFrameTime = 2e-3

        #self.do.scale = 0
        self.crosshairs = True
        #self.showSelection = True
        self.selecting = False

        self.aspect = 1.

        self.slice = None 
        
        self.overlays = []
        
        self._oldIm = None
        self._oldImSig = None
        
        self.CenteringHandlers = []
        
        self.selectHandlers = []
        
        self.labelPens = [wx.Pen(wx.Colour(*[int(c) for c in matplotlib.cm.hsv(v, alpha=.5, bytes=True)]), 2) for v in numpy.linspace(0, 1, 16)]

#        if not aspect is None:
#            if scipy.isscalar(aspect):
#                self.do.aspects[2] = aspect
#            elif len(aspect) == 3:
#                self.do.aspects = aspect

        
        #self.SetOpts()
        #self.optionspanel.RefreshHists()
        self.updating = 0
        self.showOptsPanel = 1

        self.refrTimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnRefrTimer)

        self.imagepanel.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.imagepanel.Bind(wx.EVT_KEY_DOWN, self.OnKeyPress)
        #wx.EVT_KEY_DOWN(self.Parent(), self.OnKeyPress)
        self.imagepanel.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.imagepanel.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)

        self.imagepanel.Bind(wx.EVT_MIDDLE_DOWN, self.OnMiddleDown)
        self.imagepanel.Bind(wx.EVT_MIDDLE_UP, self.OnMiddleUp)

        self.imagepanel.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)
        self.imagepanel.Bind(wx.EVT_RIGHT_UP, self.OnRightUp)

        self.imagepanel.Bind(wx.EVT_MIDDLE_DCLICK, self.OnMiddleDClick)

        self.imagepanel.Bind(wx.EVT_MOTION, self.OnMotion)

        #
        self.imagepanel.Bind(wx.EVT_ERASE_BACKGROUND, self.DoNix)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.DoNix)

    def OnRefrTimer(self, event):
        self.Refresh()
        self.Update()
        
    def SetDataStack(self, ds):
        self.do.SetDataStack(ds)
        self.SetVirtualSize(wx.Size(self.do.ds.shape[0],self.do.ds.shape[1]))
                
        self.do.xp=0
        self.do.yp=0
        self.do.zp=0
        self.do.Optimise()
            
        self.do.ResetSelection()
        
        self.Layout()
#        self.Refresh()

    def ResetDataStack(self, ds):
        self.do.SetDataStack(ds)

    def _ScreenToAbsCoordinates(self, x, y):
        xp,yp = self.CalcUnscrolledPosition(x,y)
        #xp = self.centreX + self.glCanvas.pixelsize*(x - self.Size[0]/2)
        #yp = self.centreY - self.glCanvas.pixelsize*(y - self.Size[1]/2)
        if self.do.orientation == self.do.UPRIGHT:
            return xp, yp
        else:
            return yp, xp

    def _ScreenToPixelCoordinates(self, x, y):
        #sc = pow(2.0,(self.do.scale-2))
        xp, yp = self._ScreenToAbsCoordinates(x, y)
        
        return xp/self.scale, yp/(self.scale*self.aspect)
        

    def _AbsToScreenCoordinates(self, x, y):
        x0,y0 = self.CalcUnscrolledPosition(0,0)

        if self.do.orientation == self.do.UPRIGHT:        
            return x - x0, y - y0
        else:
            return y - x0, x - y0

    def _PixelToScreenCoordinates(self, x, y):
        #sc = pow(2.0,(self.do.scale-2))
        #print self.scale, self.aspect
        return self._AbsToScreenCoordinates(x*self.scale, y*self.scale*self.aspect)
        
    def _PixelToScreenCoordinates3D(self, x, y, z):
        #sc = pow(2.0,(self.do.scale-2))
        if (self.do.slice == self.do.SLICE_XY):
            xs, ys = self._PixelToScreenCoordinates(x,y)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs, ys = self._PixelToScreenCoordinates(x,z)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs, ys = self._PixelToScreenCoordinates(y,z)
            
        return xs, ys
        
    def _drawBoxPixelCoords(self, dc, x, y, z, w, h, d):
        """Draws a box in screen space given 3D co-ordinates"""
        if (self.do.slice == self.do.SLICE_XY):
            xs, ys = self._PixelToScreenCoordinates(x,y)
            ws, hs = (w*self.scale, h*self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs, ys = self._PixelToScreenCoordinates(x,z)
            ws, hs = (w*self.scale, d*self.scale*self.aspect)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs, ys = self._PixelToScreenCoordinates(y,z)
            ws, hs = (h*self.scale, d*self.scale*self.aspect)
            
        dc.DrawRectangle(xs - 0.5*ws, ys - 0.5*hs, ws,hs)
        
        
    @property
    def scale(self):
        return pow(2.0,(self.do.scale))
        
    
    def DrawCrosshairs(self, view, dc):
        if self.crosshairs:
            sX, sY = view.imagepanel.Size
            
            dc.SetPen(wx.Pen(wx.CYAN,1))
            if(view.do.slice == view.do.SLICE_XY):
                lx = view.do.xp
                ly = view.do.yp
            elif(view.do.slice == view.do.SLICE_XZ):
                lx = view.do.xp
                ly = view.do.zp
            elif(view.do.slice == view.do.SLICE_YZ):
                lx = view.do.yp
                ly = view.do.zp
        
            
            xc, yc = view._PixelToScreenCoordinates(lx, ly)            
            dc.DrawLine(0, yc, sX, yc)
            dc.DrawLine(xc, 0, xc, sY)
            
            dc.SetPen(wx.NullPen)
            
    def DrawSelection(self, view, dc):
        if self.do.showSelection:
            col = wx.TheColourDatabase.FindColour('YELLOW')
            #col.Set(col.red, col.green, col.blue, 125)
            dc.SetPen(wx.Pen(col,1))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            lx, ly, hx, hy = self.do.GetSliceSelection()
            lx, ly = self._PixelToScreenCoordinates(lx, ly)
            hx, hy = self._PixelToScreenCoordinates(hx, hy)
            
            if self.do.selectionMode == DisplayOpts.SELECTION_RECTANGLE:
                dc.DrawRectangle(lx,ly, (hx-lx),(hy-ly))
                
            elif self.do.selectionMode == DisplayOpts.SELECTION_SQUIGGLE:
                if len(self.do.selection_trace) > 2:
                    x, y = numpy.array(self.do.selection_trace).T
                    pts = numpy.vstack(self._PixelToScreenCoordinates(x, y)).T
                    dc.DrawSpline(pts)
            elif self.do.selectionWidth == 1:
                dc.DrawLine(lx,ly, hx,hy)
            else:
                lx, ly, hx, hy = self.do.GetSliceSelection()
                dx = hx - lx
                dy = hy - ly

                if dx == 0 and dy == 0: #special case - profile is orthogonal to current plane
                    d_x = 0.5*self.do.selectionWidth
                    d_y = 0.5*self.do.selectionWidth
                else:
                    d_x = 0.5*self.do.selectionWidth*dy/numpy.sqrt((dx**2 + dy**2))
                    d_y = 0.5*self.do.selectionWidth*dx/numpy.sqrt((dx**2 + dy**2))
                    
                x_0, y_0 = self._PixelToScreenCoordinates(lx + d_x, ly - d_y)
                x_1, y_1 = self._PixelToScreenCoordinates(lx - d_x, ly + d_y)
                x_2, y_2 = self._PixelToScreenCoordinates(hx - d_x, hy + d_y)
                x_3, y_3 = self._PixelToScreenCoordinates(hx + d_x, hy - d_y)
                
                lx, ly = self._PixelToScreenCoordinates(lx, ly)
                hx, hy = self._PixelToScreenCoordinates(hx, hy)
               

                dc.DrawLine(lx, ly, hx, hy)
                dc.DrawPolygon([(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)])
                    
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)
            
    def DrawTracks(self, view, dc):
        if self.showTracks and 'filter' in dir(self) and 'clumpIndex' in self.filter.keys():
            t = self.filter['t'] #prob safe as int
            x = self.filter['x']/self.voxelsize[0]
            y = self.filter['y']/self.voxelsize[1]
            
            xb, yb, zb = self._calcVisibleBounds()
            
            IFoc = (x >= xb[0])*(y >= yb[0])*(t >= zb[0])*(x < xb[1])*(y < yb[1])*(t < zb[1])
            
            tFoc = list(set(self.filter['clumpIndex'][IFoc]))

            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            #pGreen = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #dc.SetPen(pGreen)
            

            for tN in tFoc:
                IFoc = (self.filter['clumpIndex'] == tN)
                if IFoc.sum() < 2:
                    return
                pFoc = numpy.vstack(self._PixelToScreenCoordinates3D(x[IFoc], y[IFoc], t[IFoc])).T
                
                #print pFoc.shape
                dc.SetPen(self.labelPens[tN%16])
                dc.DrawLines(pFoc)
                
    def DrawScaleBar(self, view, dc):
        if self.showScaleBar:
            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('WHITE'),10)
            pGreen.SetCap(wx.CAP_BUTT)
            dc.SetPen(pGreen)
            sX, sY = view.imagepanel.Size
            
            sbLen = self.scaleBarLength*view.scale/view.voxelsize[0]
            
            y1 = 20
            x1 = 20 + sbLen
            x0 = x1 - sbLen
            dc.DrawLine(x0, y1, x1, y1)
            
            dc.SetTextForeground(wx.TheColourDatabase.FindColour('WHITE'))
            if self.scaleBarLength > 1000:
                s = u'%1.1f \u00B5m' % (self.scaleBarLength / 1000.)
            else:
                s = u'%d nm' % int(self.scaleBarLength)
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, x0 + (sbLen - w)/2, y1 + 7)
                
    def DrawContours(self, view, dc):
        if self.showContours and 'filter' in dir(self) and 'contour' in self.filter.keys() and self.do.slice ==self.do.SLICE_XY:
            t = self.filter['t'] # prob safe as int
            x = self.filter['x']/self.voxelsize[0]
            y = self.filter['y']/self.voxelsize[1]
            
            xb, yb, zb = self._calcVisibleBounds()
            
            IFoc = (x >= xb[0])*(y >= yb[0])*(t >= zb[0])*(x < xb[1])*(y < yb[1])*(t < zb[1])

            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            #pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            dc.SetPen(pGreen)
            
            contours = self.filter['contour'][IFoc]
            if 'clumpIndex' in self.filter.keys():
                colInds = self.filter['clumpIndex'][IFoc] %len(self.labelPens)
            else:
                colInds = numpy.zeros(len(contours), 'i') #%len(self.labelPens)
            for c, colI in zip(contours, colInds):
                xc, yc = c.T
                dc.SetPen(self.labelPens[int(colI)])
                dc.DrawSpline(numpy.vstack(self._PixelToScreenCoordinates(xc, yc)).T)
                
    
       
    def DrawPoints(self, view, dc):
        dx = 0
        dy = 0
        
        aN = SLICE_AXIS_LUT[self.do.slice]
        tolN = TOL_AXIS_LUT[self.do.slice]
        pos = [self.do.xp, self.do.yp, self.do.zp]

        if self.showPoints and ('filter' in dir(self) or len(self.points) > 0):
            if 'filter' in dir(self):
                t = self.filter['t'] #prob safe as int
                x = self.filter['x']/self.voxelsize[0]
                y = self.filter['y']/self.voxelsize[1]
                
                xb, yb, zb = self._calcVisibleBounds()
                
                IFoc = (x >= xb[0])*(y >= yb[0])*(t >= zb[0])*(x < xb[1])*(y < yb[1])*(t < zb[1])
                    
                pFoc = numpy.vstack((x[IFoc], y[IFoc], t[IFoc])).T
                if self.pointMode == 'splitter':
                    pCol = self.filter['gFrac'][IFoc] > .5                
                pNFoc = []

            #intrinsic points            
            elif len(self.points) > 0:
                pointTol = self.pointTolNFoc[self.pointMode]
                
                IFoc = abs(self.points[:,aN] - pos[aN]) < 1
                INFoc = abs(self.points[:,aN] - pos[aN]) < pointTol[tolN]
                    
                pFoc = self.points[IFoc]
                pNFoc = self.points[INFoc]
                
                if self.pointMode == 'splitter':
                    pCol = self.pointColours[IFoc]
                    
            if self.pointMode == 'splitter':
                if 'chroma' in dir(self):
                    dx = self.chroma.dx.ev(pFoc[:,0]*1e3*self.voxelsize[0], pFoc[:,1]*1e3*self.voxelsize[1])/(1e3*self.voxelsize[0])
                    dy = self.chroma.dy.ev(pFoc[:,0]*1e3*self.voxelsize[0], pFoc[:,1]*1e3*self.voxelsize[1])/(1e3*self.voxelsize[1])
                else:
                    dx = 0*pFoc[:,0]
                    dy = 0*pFoc[:,0]

                if 'chroma' in dir(self):
                    dxn = self.chroma.dx.ev(pNFoc[:,0]*1e3*self.voxelsize[0], pNFoc[:,1]*1e3*self.voxelsize[1])/(1e3*self.voxelsize[0])
                    dyn = self.chroma.dy.ev(pNFoc[:,0]*1e3*self.voxelsize[0], pNFoc[:,1]*1e3*self.voxelsize[1])/(1e3*self.voxelsize[1])
                else:
                    dxn = 0*pNFoc[:,0]
                    dyn = 0*pNFoc[:,0]

            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            ps = self.pointSize

            if self.showAdjacentPoints:
                dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('BLUE'),1))
                
                if self.pointMode == 'splitter':
                    for p, dxi, dyi in zip(pNFoc, dxn, dyn):
                        self._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
                        self._drawBoxPixelCoords(dc, p[0]-dxi, 0.5*self.do.ds.shape[1] + p[1]-dyi, p[2], ps, ps, ps)

                else:
                    for p in pNFoc:
                        self._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)


            pGreen = wx.Pen(wx.TheColourDatabase.FindColour('GREEN'),1)
            pRed = wx.Pen(wx.TheColourDatabase.FindColour('RED'),1)
            dc.SetPen(pGreen)
            
            if self.pointMode == 'splitter':
                for p, c, dxi, dyi in zip(pFoc, pCol, dx, dy):
                    if c:
                        dc.SetPen(pGreen)
                    else:
                        dc.SetPen(pRed)
                        
                    self._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
                    self._drawBoxPixelCoords(dc, p[0]-dxi, 0.5*self.do.ds.shape[1] + p[1]-dyi, p[2], ps, ps, ps)
                    
            else:
                for p in pFoc:
                    self._drawBoxPixelCoords(dc, p[0], p[1], p[2], ps, ps, ps)
            
            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)

    def _calcVisibleBounds(self):
        sc = pow(2.0,(self.do.scale)) 
        x0,y0 = self.CalcUnscrolledPosition(0,0)
        sX, sY = self.imagepanel.Size
        
        if self.do.slice == self.do.SLICE_XY:
            bnds = [(x0/sc, (x0+sX)/sc), (y0/sc, (y0+sY)/sc), (self.do.zp-.5, self.do.zp+.5)]
        elif self.do.slice == self.do.SLICE_XZ:
            bnds = [(x0/sc, (x0+sX)/sc), (self.do.yp-.5, self.do.yp+.5), (y0/sc, (y0+sY)/sc)]
        elif self.do.slice == self.do.SLICE_YZ:
            bnds = [(self.do.xp-.5, self.do.xp+.5),(x0/sc, (x0+sX)/sc), (y0/sc, (y0+sY)/sc)]

        return bnds
        
    
    
    def DoPaint(self, dc, fullImage=False):
        #print 'p'
        
        dc.Clear()
                                     
        im = self.Render(fullImage)

        sc = pow(2.0,(self.do.scale))
        sc2 = sc
        
        if sc >= 1:
            step = 1
        else:
            step = 2**(-numpy.ceil(numpy.log2(sc)))
            sc2 = sc*step
            
        #sX, sY = view.imagepanel.Size
            
        #im.Rescale(im.GetWidth()*sc2,im.GetHeight()*sc2*self.aspect)

        
        im2 = wx.BitmapFromImage(im)
        dc.DrawBitmap(im2,-sc2/2,-sc2/2)
        
        #sX, sY = im.GetWidth(), im.GetHeight()

        self.DrawCrosshairs(self, dc)
        self.DrawSelection(self, dc) 
        self.DrawScaleBar(self, dc)

        #self.DrawTracks(self, dc)
        self.DrawPoints(self, dc)
        self.DrawContours(self, dc)

        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
            
        for ovl in self.do.overlays:
            ovl(self, dc)

    def GrabImage(self, fullImage=True):
        #TODO - get suitable image dependent viewport

        xs, ys = self._unscrolled_view_size()
        if fullImage:
            if (xs > 2e3 or ys > 2e3) and wx.MessageBox('Captured image will be very large, continue?', 'Warning', style=wx.OK|wx.CANCEL) != wx.OK:
                return
        else:
            s = self.GetClientSize()
            xs = min(s.GetWidth(), xs)
            ys = min(s.GetHeight(), ys)

        MemBitmap = wx.EmptyBitmap(xs, ys)
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)

        self.DoPaint(MemDC, fullImage)

        return MemBitmap

    def GrabPNG(self, filename, fullImage=True):
        MemBitmap = self.GrabImage(fullImage)
        img = MemBitmap.ConvertToImage()
        img.SaveFile(filename, wx.BITMAP_TYPE_PNG)
        
    def GrabPNGToBuffer(self, fullImage=True):
        '''Get PNG data in a buffer (rather than writing directly to file)'''
        from io import BytesIO

        img = self.GrabImage(fullImage)
        out = BytesIO()
        # NB - using wx functionality rather than pillow here as wxImage.GetData() returns a BytesArray object rather
        # than a buffer on py3. This underlying problem may need to be revisited.
        img.ConvertToImage().SaveFile(out, wx.BITMAP_TYPE_PNG)
        return out.getvalue()

    def CopyImage(self, fullImage=True):
        """ Copies the currently displayed image to the clipboard"""
        bmp = self.GrabImage(fullImage)
        try:
            wx.TheClipboard.Open()
            bmpDataObject = wx.BitmapDataObject(bmp)
            wx.TheClipboard.SetData(bmpDataObject)
        finally:
            wx.TheClipboard.Close()

            
#    def OnPaint(self,event):
#        self.painting = True
#        DC = wx.PaintDC(self.imagepanel)
#        if not time.time() > (self.lastUpdateTime + 2*self.lastFrameTime): #avoid paint floods
#            if not self.refrTimer.IsRunning():
#                self.refrTimer.Start(.2, True) #make sure we do get a refresh after disposing of flood
#            return
#
#        frameStartTime = time.time()
#        self.imagepanel.impanel.PrepareDC(DC)
#
#        x0,y0 = self.imagepanel.CalcUnscrolledPosition(0,0)
#
#        #s = self.imagepanel.GetVirtualSize()
#        s = self.imagepanel.impanel.GetClientSize()
#        MemBitmap = wx.EmptyBitmap(s.GetWidth(), s.GetHeight())
#        #del DC
#        MemDC = wx.MemoryDC()
#        OldBitmap = MemDC.SelectObject(MemBitmap)
#        try:
#            DC.BeginDrawing()
#            #DC.Clear()
#            #Perform(WM_ERASEBKGND, MemDC, MemDC);
#            #Message.DC := MemDC;
#            self.DoPaint(MemDC);
#            #Message.DC := 0;
#            #DC.BlitXY(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
#            DC.Blit(x0, y0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
#            DC.EndDrawing()
#        finally:
#            #MemDC.SelectObject(OldBitmap)
#            del MemDC
#            del MemBitmap
#
#        self.lastUpdateTime = time.time()
#        self.lastFrameTime = self.lastUpdateTime - frameStartTime
#
#        self.painting = False
#        #print self.lastFrameTime
            

    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        if rot < 0:
            if event.RightIsDown():
                self.do.yp = max(self.do.yp - 1, 0)
            elif event.MiddleIsDown(): 
                self.do.xp = max(self.do.xp - 1, 0)
            elif event.ShiftDown():
                self.do.SetScale(self.do.scale - 1)
            else:
                self.do.zp = max(self.do.zp - 1, 0)
        if rot > 0:
            if event.RightIsDown():
                self.do.yp = min(self.do.yp + 1, self.do.ds.shape[1] -1)
            elif event.MiddleIsDown(): 
                self.do.xp = min(self.do.xp + 1, self.do.ds.shape[0] -1)
            elif event.ShiftDown():
                self.do.SetScale(self.do.scale + 1)
            else:
                self.do.zp = min(self.do.zp + 1, self.do.ds.shape[2] -1)
                
        if ('update' in dir(self.GetParent())):
             self.GetParent().update()
        else:
            self.imagepanel.Refresh()
        #self.update()
    
    def OnKeyPress(self, event):
        if event.GetKeyCode() == wx.WXK_PAGEUP:
            self.do.zp = max(0, self.do.zp - 1)
            #self.optionspanel.RefreshHists()
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                #if not self.painting:
                self.imagepanel.Refresh()
                #else:
                #    if not self.refrTimer.IsRunning():
                #        self.refrTimer.Start(.2, True)

        elif event.GetKeyCode() == wx.WXK_PAGEDOWN:
            self.do.zp = min(self.do.zp + 1, self.do.ds.shape[2] - 1)
            #self.optionspanel.RefreshHists()
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
                #print 'upd'
            else:
                #if not self.painting:
                self.imagepanel.Refresh()
                #else:
                #    if not self.refrTimer.IsRunning():
                        #print 'upt'
                #        self.refrTimer.Start(.2, True)
                
        elif event.GetKeyCode() == 74: #J
            self.do.xp = (self.do.xp - 1)
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 76: #L
            self.do.xp +=1
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 73: #I
            self.do.yp -= 1
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 75: #K
            self.do.yp += 1
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 77: #M
            #print 'o'
            self.do.Optimise()
        elif event.GetKeyCode() == ord('C'):
            if event.GetModifiers() == wx.MOD_CMD:
                self.CopyImage()
            elif event.GetModifiers() == wx.MOD_CMD|wx.MOD_SHIFT:
                self.CopyImage(False)
            else:
                event.Skip()
        else:
            event.Skip()
        

        
    def GetOpts(self,event=None):
        if (self.updating == 0):

            sc = pow(2.0,(self.do.scale))
            s = self.CalcImSize()
            self.SetVirtualSize(wx.Size(s[0]*sc,s[1]*sc))

            if not self.slice == self.do.slice:
                #if the slice has changed, change our aspect and do some
                self.slice = self.do.slice
                #if not event is None and event.GetId() in [self.cbSlice.GetId(), self.cbScale.GetId()]:
                #recenter the view
                if(self.do.slice == self.do.SLICE_XY):
                    lx = self.do.xp
                    ly = self.do.yp
                    self.aspect = self.do.aspect[1]/self.do.aspect[0]
                elif(self.do.slice == self.do.SLICE_XZ):
                    lx = self.do.xp
                    ly = self.do.zp
                    self.aspect = self.do.aspect[2]/self.do.aspect[0]
                elif(self.do.slice == self.do.SLICE_YZ):
                    lx = self.do.yp
                    ly = self.do.zp
                    self.aspect = self.do.aspect[2]/self.do.aspect[1]

                sx,sy =self.imagepanel.GetClientSize()

                #self.imagepanel.SetScrollbars(20,20,s[0]*sc/20,s[1]*sc/20,min(0, lx*sc - sx/2)/20, min(0,ly*sc - sy/2)/20)
                ppux, ppuy = self.GetScrollPixelsPerUnit()
                #self.imagepanel.SetScrollPos(wx.HORIZONTAL, max(0, lx*sc - sx/2)/ppux)
                #self.imagepanel.SetScrollPos(wx.VERTICAL, max(0, ly*sc - sy/2)/ppuy)
                self.Scroll(max(0, lx*sc - sx/2)/ppux, max(0, ly*sc*self.aspect - sy/2)/ppuy)

            #self.imagepanel.Refresh()
            self.Refresh()
            self.Update()
            
    def Optim(self, event = None):
        self.do.Optimise(self.do.ds, int(self.do.zp))
        self.updating=1
        #self.SetOpts()
        #self.optionspanel.RefreshHists()
        self.Refresh()
        self.Update()
        self.updating=0
        
    def CalcImSize(self):
        if (self.do.slice == self.do.SLICE_XY):
            if (self.do.orientation == self.do.UPRIGHT):
                return (self.do.ds.shape[0],self.do.ds.shape[1])
            else:
                return (self.do.ds.shape[1],self.do.ds.shape[0])
        elif (self.do.slice == self.do.SLICE_XZ):
            return (self.do.ds.shape[0],self.do.ds.shape[2])
        else:
            return(self.do.ds.shape[1],self.do.ds.shape[2] )
        
    def DoNix(self, event):
        pass

    def OnLeftDown(self,event):
        if self.do.leftButtonAction == self.do.ACTION_SELECTION:
            self.StartSelection(event)
            
        event.Skip()
    
    def OnLeftUp(self,event):
        if self.do.leftButtonAction == self.do.ACTION_SELECTION:
            self.ProgressSelection(event)
            self.EndSelection()
        else:
            self.OnSetPosition(event)
            
        event.Skip()
        
    def OnMiddleDown(self,event):
        dc = wx.ClientDC(self.imagepanel)
#        self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        self.middleDownPos = self.CalcUnscrolledPosition(*pos)
        event.Skip()
    
    def OnMiddleUp(self,event):
        dc = wx.ClientDC(self.imagepanel)
#        self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)

        dx = pos[0] - self.middleDownPos[0]
        dy = pos[1] - self.middleDownPos[1]
        
        sc = pow(2.0,(self.do.scale))

        if (abs(dx) > 5) or (abs(dy) > 5):
            for h in self.CenteringHandlers:
                h(-dx/sc,-dy/sc)
        
        event.Skip()
        
    def OnMiddleDClick(self,event):
        dc = wx.ClientDC(self.imagepanel)
#        self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)
        #print pos
        sc = pow(2.0,(self.do.scale))
        if (self.do.slice == self.do.SLICE_XY):
            x = (pos[0]/sc) - 0.5*self.do.ds.shape[0]
            y = (pos[1]/(sc*self.aspect)) - 0.5*self.do.ds.shape[1]
            
            for h in self.CenteringHandlers:
                h(x,y)
            
        event.Skip()
    
            
    def _unscrolled_view_size(self):
        sc = pow(2.0, (self.do.scale))
        shp = self.do.ds.shape

        if (self.do.slice == self.do.SLICE_XY):
            xs = int(shp[0] * sc)
            ys = int(shp[1] * sc*self.aspect)
        elif (self.do.slice == self.do.SLICE_XZ):
            xs = int(shp[0] * sc)
            ys = int(shp[2] * sc * self.aspect)
        elif (self.do.slice == self.do.SLICE_YZ):
            xs = int(shp[1] * sc)
            ys = int(shp[2] * sc * self.aspect)

        return xs, ys

    def OnSetPosition(self,event):
        dc = wx.ClientDC(self.imagepanel)
        #self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)

        #print(pos)
        self.do.inOnChange = True
        try:
            sc = pow(2.0,(self.do.scale))
            #print(sc)
            if (self.do.slice == self.do.SLICE_XY):
                self.do.xp =int(pos[0]/sc)
                self.do.yp = int(pos[1]/(sc*self.aspect))
            elif (self.do.slice == self.do.SLICE_XZ):
                self.do.xp =int(pos[0]/sc)
                self.do.zp =int(pos[1]/(sc*self.aspect))
            elif (self.do.slice == self.do.SLICE_YZ):
                self.do.yp =int(pos[0]/sc)
                self.do.zp =int(pos[1]/(sc*self.aspect))
        finally:
            self.do.inOnChange = False
            
        self.do.OnChange()
        
        for cb in self.selectHandlers:
            cb(self)
        #if ('update' in dir(self.GetParent())):
        #     self.GetParent().update()
        #else:
        #    self.imagepanel.Refresh()

    def PointsHitTest(self):
        if len(self.points) > 0:
            iCand = numpy.where((abs(self.points[:,2] - self.do.zp) < 1)*(abs(self.points[:,0] - self.do.xp) < 3)*(abs(self.points[:,1] - self.do.yp) < 3))[0]

            if len(iCand) == 0:
                return None
            elif len(iCand) == 1:
                return iCand[0]
            else:
                pCand = self.points[iCand, :]

                iNearest = numpy.argmin((pCand[:,0] - self.do.xp)**2 + (pCand[:,1] - self.do.yp)**2)

                return iCand[iNearest]
        else:
            return None


    def OnRightDown(self, event):
        self.StartSelection(event)
            
    def StartSelection(self,event):
        self.selecting = True
        
        dc = wx.ClientDC(self.imagepanel)
        #self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)
        #print pos
        sc = pow(2.0,(self.do.scale))
        if (self.do.slice == self.do.SLICE_XY):
            self.do.selection_begin_x = int(pos[0]/sc)
            self.do.selection_begin_y = int(pos[1]/(sc*self.aspect))
        elif (self.do.slice == self.do.SLICE_XZ):
            self.do.selection_begin_x = int(pos[0]/sc)
            self.do.selection_begin_z = int(pos[1]/(sc*self.aspect))
        elif (self.do.slice == self.do.SLICE_YZ):
            self.do.selection_begin_y = int(pos[0]/sc)
            self.do.selection_begin_z = int(pos[1]/(sc*self.aspect))
            
        self.do.selection_trace = []
        self.do.selection_trace.append(((pos[0]/sc), (pos[1]/(sc*self.aspect))))

    def OnRightUp(self,event):
        self.ProgressSelection(event)
        self.EndSelection()

    def OnMotion(self, event):
        if event.Dragging() and self.selecting:
            self.ProgressSelection(event)
            
    def ProgressSelection(self,event):
        dc = wx.ClientDC(self.imagepanel)
        #self.imagepanel.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)
        #print pos
        
        sc = pow(2.0,(self.do.scale))

        if not event.ShiftDown():
            if (self.do.slice == self.do.SLICE_XY):
                self.do.selection_end_x = int(pos[0]/sc)
                self.do.selection_end_y = int(pos[1]/(sc*self.aspect))
            elif (self.do.slice == self.do.SLICE_XZ):
                self.do.selection_end_x = int(pos[0]/sc)
                self.do.selection_end_z = int(pos[1]/(sc*self.aspect))
            elif (self.do.slice == self.do.SLICE_YZ):
                self.do.selection_end_y = int(pos[0]/sc)
                self.do.selection_end_z = int(pos[1]/(sc*self.aspect))
        else: #lock
            if (self.do.slice == self.do.SLICE_XY):
                self.do.selection_end_x = int(pos[0]/sc)
                self.do.selection_end_y = int(pos[1]/(sc*self.aspect))

                dx = abs(self.do.selection_end_x - self.do.selection_begin_x)
                dy = abs(self.do.selection_end_y - self.do.selection_begin_y)

                if dx > 1.5*dy: #horizontal
                    self.do.selection_end_y = self.do.selection_begin_y
                elif dy > 1.5*dx: #vertical
                    self.do.selection_end_x = self.do.selection_begin_x
                else: #diagonal
                    self.do.selection_end_y = self.do.selection_begin_y + dx*numpy.sign(self.do.selection_end_y - self.do.selection_begin_y)

            elif (self.do.slice == self.do.SLICE_XZ):
                self.do.selection_end_x = int(pos[0]/sc)
                self.do.selection_end_z = int(pos[1]/(sc*self.aspect))
            elif (self.do.slice == self.do.SLICE_YZ):
                self.do.selection_end_y = int(pos[0]/sc)
                self.do.selection_end_z = int(pos[1]/(sc*self.aspect))
                
        self.do.selection_trace.append(((pos[0]/sc), (pos[1]/(sc*self.aspect))))

        #if ('update' in dir(self.GetParent())):
        #     self.GetParent().update()
        #self.update()
        #else:
        self.Refresh()
        self.Update()

    def EndSelection(self):
        self.selecting = False
        self.do.EndSelection()
            

        
#    def Render(self):
#        x0,y0 = self.imagepanel.CalcUnscrolledPosition(0,0)
#        sX, sY = self.imagepanel.Size
#
#        sc = pow(2.0,(self.do.scale-2))
#        sX_ = int(sX/sc)
#        sY_ = int(sY/sc)
#        x0_ = int(x0/sc)
#        y0_ = int(y0/sc)
#
#        #XY
#        if self.do.slice == DisplayOpts.SLICE_XY:
#            if self.do.Chans[0] < self.do.ds.shape[3]:
#                r = (self.do.Gains[0]*(self.do.ds[x0_:(x0_+sX_),y0_:(y0_+sY_),int(self.do.zp), self.do.Chans[0]] - self.do.Offs[0])).astype('uint8').squeeze().T
#            else:
#                r = numpy.zeros(ds.shape[:2], 'uint8').T
#            if self.do.Chans[1] < self.do.ds.shape[3]:
#                g = (self.do.Gains[1]*(self.do.ds[x0_:(x0_+sX_),y0_:(y0_+sY_),int(self.do.zp), self.do.Chans[1]] - self.do.Offs[1])).astype('uint8').squeeze().T
#            else:
#                g = numpy.zeros(ds.shape[:2], 'uint8').T
#            if self.do.Chans[2] < self.do.ds.shape[3]:
#                b = (self.do.Gains[2]*(self.do.ds[x0_:(x0_+sX_),y0_:(y0_+sY_),int(self.do.zp), self.do.Chans[2]] - self.do.Offs[2])).astype('uint8').squeeze().T
#            else:
#                b = numpy.zeros(ds.shape[:2], 'uint8').T
#        #XZ
#        elif self.do.slice == DisplayOpts.SLICE_XZ:
#            if self.do.Chans[0] < self.do.ds.shape[3]:
#                r = (self.do.Gains[0]*(self.do.ds[x0_:(x0_+sX_),int(self.do.yp),y0_:(y0_+sY_), self.do.Chans[0]] - self.do.Offs[0])).astype('uint8').squeeze().T
#            else:
#                r = numpy.zeros((ds.shape[0], ds.shape[2]), 'uint8').T
#            if self.do.Chans[1] < self.do.ds.shape[3]:
#                g = (self.do.Gains[1]*(self.do.ds[x0_:(x0_+sX_),int(self.do.yp),y0_:(y0_+sY_), self.do.Chans[1]] - self.do.Offs[1])).astype('uint8').squeeze().T
#            else:
#                g = numpy.zeros((ds.shape[0], ds.shape[2]), 'uint8').T
#            if self.do.Chans[2] < self.do.ds.shape[3]:
#                b = (self.do.Gains[2]*(self.do.ds[x0_:(x0_+sX_),int(self.do.yp),y0_:(y0_+sY_), self.do.Chans[2]] - self.do.Offs[2])).astype('uint8').squeeze().T
#            else:
#                b = numpy.zeros((ds.shape[0], ds.shape[2]), 'uint8'.T)
#
#        #YZ
#        elif self.do.slice == DisplayOpts.SLICE_YZ:
#            if self.do.Chans[0] < self.do.ds.shape[3]:
#                r = (self.do.Gains[0]*(self.do.ds[int(self.do.xp),x0_:(x0_+sX_),y0_:(y0_+sY_), self.do.Chans[0]] - self.do.Offs[0])).astype('uint8').squeeze().T
#            else:
#                r = numpy.zeros((ds.shape[1], ds.shape[2]), 'uint8').T
#            if self.do.Chans[1] < self.do.ds.shape[3]:
#                g = (self.do.Gains[1]*(self.do.ds[int(self.do.xp),x0_:(x0_+sX_),y0_:(y0_+sY_), self.do.Chans[1]] - self.do.Offs[1])).astype('uint8').squeeze().T
#            else:
#                g = numpy.zeros((ds.shape[1], ds.shape[2]), 'uint8').T
#            if self.do.Chans[2] < self.do.ds.shape[3]:
#                b = (self.do.Gains[2]*(self.do.ds[int(self.do.xp),x0_:(x0_+sX_),y0_:(y0_+sY_), self.do.Chans[2]] - self.do.Offs[2])).astype('uint8').squeeze().T
#            else:
#                b = numpy.zeros((ds.shape[1], ds.shape[2]), 'uint8'.T)
#        r = r.T
#        g = g.T
#        b = b.T
#        r = r.reshape(r.shape + (1,))
#        g = g.reshape(g.shape + (1,))
#        b = b.reshape(b.shape + (1,))
#        ima = numpy.concatenate((r,g,b), 2)
#        return wx.ImageFromData(ima.shape[1], ima.shape[0], ima.ravel())
        
    def _gensig(self, x0, y0, sX,sY, do):
        sig = [x0, y0, sX, sY, do.scale, do.slice, do.GetActiveChans(), do.ds.shape]
        if do.slice == DisplayOpts.SLICE_XY:
            sig += [do.zp, do.maximumProjection]
        if do.slice == DisplayOpts.SLICE_XZ:
            sig += [do.yp]
        if do.slice == DisplayOpts.SLICE_YZ:
            sig += [do.xp]
            
        return sig
    
    def Redraw(self, sender=None, **kwargs):
        self._oldImSig = None
        self.Refresh()
        self.Update()

    def Render(self, fullImage=False):
        #print 'rend'
        if fullImage:
            x0, y0 = 0,0
            sX, sY = self._unscrolled_view_size()
        else:
            x0,y0 = self.CalcUnscrolledPosition(0,0)
            sX, sY = self.imagepanel.Size
        
        sig = self._gensig(x0, y0, sX, sY, self.do)
        if sig == self._oldImSig:# and not self._oldIm is None:
            #if nothing has changed, don't re-render
            return self._oldIm

        sc = pow(2.0,self.do.scale)
        sc2 = sc

        if sc >= 1:
            step = 1
        else:
            step = 2**(-numpy.ceil(numpy.log2(sc)))
            sc2 = sc*step
        
        sX_ = int(sX/(sc))
        sY_ = int(sY/(sc*self.aspect))
        x0_ = int(x0/sc)
        y0_ = int(y0/(sc*self.aspect))
            
        fstep = float(step)
        step = int(step)

        #XY
        if self.do.slice == DisplayOpts.SLICE_XY:
            ima = numpy.zeros((int(numpy.ceil(min(sY_, self.do.ds.shape[1])/fstep)), int(numpy.ceil(min(sX_, self.do.ds.shape[0])/fstep)), 3), 'uint8')
            for chan, offset, gain, cmap in self.do.GetActiveChans():
                if not cmap == labeled:
                    lut = getLUT(cmap)
                    
                    if self.do.maximumProjection:
                        seg = self.do.ds[x0_:(x0_+sX_):step,y0_:(y0_+sY_):step,:, chan].max(2).squeeze().T
                        if self.do.colourMax:
                            aseg = self.do.ds[x0_:(x0_+sX_):step,y0_:(y0_+sY_):step,:, chan].argmax(2).squeeze().T
                            applyLUT(aseg, self.do.cmax_scale/self.do.ds.shape[2], self.do.cmax_offset, lut, ima)
                            ima[:] = (ima*numpy.clip((seg - offset)*gain, 0,1)[:,:,None]).astype('uint8')
                        else:
                            applyLUT(seg, gain, offset, lut, ima)
                    else:
                        seg = self.do.ds[x0_:(x0_+sX_):step,y0_:(y0_+sY_):step,int(self.do.zp), chan].squeeze().T
                        
                        if numpy.iscomplexobj(seg):
                            if self.do.complexMode == 'real':
                                applyLUT(seg.real, gain, offset, lut, ima)
                            elif self.do.complexMode == 'imag':
                                applyLUT(seg.imag, gain, offset, lut, ima)
                            elif self.do.complexMode == 'abs':
                                applyLUT(numpy.abs(seg), gain, offset, lut, ima)
                            elif self.do.complexMode == 'angle':
                                applyLUT(numpy.angle(seg), gain, offset, lut, ima)
                            else:
                                applyLUT(numpy.angle(seg), self.do.cmax_scale/self.do.ds.shape[2], self.do.cmax_offset, lut, ima)
                                ima[:] = (ima*numpy.clip((numpy.abs(seg) - offset)*gain, 0,1)[:,:,None]).astype('uint8')
                        else:
                            #print seg.shape
                            applyLUT(seg, gain, offset, lut, ima)

                else:
                    if self.layerMode == 'mult':
                        ima[:] = numpy.minimum(ima[:]*(cmap(gain*(self.do.ds[x0_:(x0_+sX_):step,y0_:(y0_+sY_):step,int(self.do.zp), chan].squeeze().T - offset))[:,:,:3])[:], 255)
                    else:
                        ima[:] = numpy.minimum(ima[:] + (255*cmap(gain*(self.do.ds[x0_:(x0_+sX_):step,y0_:(y0_+sY_):step,int(self.do.zp), chan].squeeze().T - offset))[:,:,:3])[:], 255)
        #XZ
        elif self.do.slice == DisplayOpts.SLICE_XZ:
            ima = numpy.zeros((int(numpy.ceil(min(sY_, self.do.ds.shape[2])/fstep)), int(numpy.ceil(min(sX_, self.do.ds.shape[0])/fstep)), 3), 'uint8')

            for chan, offset, gain, cmap in self.do.GetActiveChans():#in zip(self.do.Chans, self.do.Offs, self.do.Gains, self.do.cmaps):
                if not cmap == labeled:
                    lut = getLUT(cmap)
                    seg = self.do.ds[x0_:(x0_+sX_):step,int(self.do.yp),y0_:(y0_+sY_):step, chan].squeeze().T
                    applyLUT(seg, gain, offset, lut, ima)
                else:
                    ima[:] = ima[:] + 255*cmap(gain*(self.do.ds[x0_:(x0_+sX_):step,int(self.do.yp),y0_:(y0_+sY_):step, chan].squeeze().T - offset))[:,:,:3][:]

        #YZ
        elif self.do.slice == DisplayOpts.SLICE_YZ:
            ima = numpy.zeros((int(numpy.ceil(min(sY_, self.do.ds.shape[2])/fstep)), int(numpy.ceil(min(sX_, self.do.ds.shape[1])/fstep)), 3), 'uint8')

            for chan, offset, gain, cmap in self.do.GetActiveChans():#zip(self.do.Chans, self.do.Offs, self.do.Gains, self.do.cmaps):
                if not cmap == labeled:
                    lut = getLUT(cmap)
                    seg = self.do.ds[int(self.do.xp),x0_:(x0_+sX_):step,y0_:(y0_+sY_):step, chan].squeeze().T
                    applyLUT(seg, gain, offset, lut, ima)
                else:
                    ima[:] = ima[:] + 255*cmap(gain*(self.do.ds[int(self.do.xp),x0_:(x0_+sX_):step,y0_:(y0_+sY_):step, chan].squeeze().T - offset))[:,:,:3][:]
#        
        img = wx.ImageFromData(ima.shape[1], ima.shape[0], ima.ravel())
        img.Rescale(img.GetWidth()*sc2,img.GetHeight()*sc2*self.aspect)
        self._oldIm = img
        self._oldImSig = sig
        return img


#    def GetProfile(self,halfLength=10,axis = 2, pos=None, roi=[2,2], background=None):
#        if not pos is None:
#            px, py, pz = pos
#        else:
#            px, py, pz = self.do.xp, self.do.yp, self.do.zp
#
#        points = self.points
#        d = None
#        pts = None
#
#        if axis == 2: #z
#            p = self.do.ds[(px - roi[0]):(px + roi[0]),(py - roi[1]):(py + roi[1]),(pz - halfLength):(pz + halfLength)].mean(2).mean(1)
#            x = numpy.mgrid[(pz - halfLength):(pz + halfLength)]
#            if len(points) > 0:
#                d = numpy.array([((abs(points[:,0] - px) < 2*roi[0])*(abs(points[:,1] - py) < 2*roi[1])*(points[:,2] == z)).sum() for z in x])
#
#                pts = numpy.where((abs(points[:,0] - px) < 2*roi[0])*(abs(points[:,1] - py) < 2*roi[1])*(abs(points[:,2] - pz) < halfLength))
#            #print p.shape
#            #p = p.mean(1).mean(0)
#            if not background is None:
#                p -= self.do.ds[(px - background[0]):(px + background[0]),(py - background[1]):(py + background[1]),(pz - halfLength):(pz + halfLength)].mean(2).mean(1)
#        elif axis == 1: #y
#            p = self.do.ds[(px - roi[0]):(px + roi[0]),(py - halfLength):(py + halfLength),(pz - roi[1]):(pz + roi[1])].mean(1).mean(0)
#            x = numpy.mgrid[(py - halfLength):(py + halfLength)]
#            if len(points) > 0:
#                d = numpy.array([((abs(points[:,1] - py) < 2*roi[0])*(abs(points[:,2] - pz) < 2*roi[1])*(points[:,0] == z)).sum() for z in x])
#
#                pts = numpy.where((abs(points[:,0] - px) < 2*roi[0])*(abs(points[:,1] - py) < halfLength)*(abs(points[:,2] - pz) < 2*roi[1]))
#            if not background is None:
#                p -= self.do.ds[(px - background[0]):(px + background[0]),(py - halfLength):(py + halfLength),(pz - background[1]):(pz + background[1]),(pz - halfLength):(pz + halfLength)].mean(1).mean(0)
#        elif axis == 0: #x
#            p = self.do.ds[(px - halfLength):(px + halfLength), (py - roi[0]):(py + roi[0]),(pz - roi[1]):(pz + roi[1])].mean(2).mean(0)
#            x = numpy.mgrid[(px - halfLength):(px + halfLength)]
#            if len(points) > 0:
#                d = numpy.array([((abs(points[:,0] - px) < 2*roi[0])*(abs(points[:,2] - pz) < 2*roi[1])*(points[:,1] == z)).sum() for z in x])
#
#                pts = numpy.where((abs(points[:,0] - px) < halfLength)*(abs(points[:,1] - py) < 2*roi[0])*(abs(points[:,2] - pz) < 2*roi[1]))
#            if not background is None:
#                p -= self.do.ds[(px - halfLength):(px + halfLength),(py - background[0]):(py + background[0]),(pz - background[1]):(pz + background[1])].mean(2).mean(0)
#
#        return x,p,d, pts
# end of class ViewPanel

class ArraySettingsAndViewPanel(wx.Panel):
    def __init__(self, parent, dstack = None, aspect=1, horizOptions = False, wantUpdates = [], mdh=None, **kwds):
        kwds["style"] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent, **kwds)
        self.showOptsPanel = 1

        self.WantUpdateNotification = []
        self.WantUpdateNotification += wantUpdates

        self._mgr = aui.AuiManager(agwFlags = aui.AUI_MGR_DEFAULT)
        # tell AuiManager to manage this frame
        self._mgr.SetManagedWindow(self)

        self.view = ArrayViewPanel(self, dstack, aspect)
        self._mgr.AddPane(self.view, aui.AuiPaneInfo().
                          Name("Data").Caption("Data").Centre().CloseButton(False).CaptionVisible(False))

        self.do = self.view.do

        self.optionspanel = OptionsPanel(self, self.view.do,horizOrientation = horizOptions)
        self.optionspanel.SetSize(self.optionspanel.GetBestSize())
        pinfo = aui.AuiPaneInfo().Name("optionsPanel").Right().Caption('Display Settings').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        self._mgr.AddPane(self.optionspanel, pinfo)

        self.overlaypanel = OverlayPanel(self, self.view, mdh)
        self.overlaypanel.SetSize(self.overlaypanel.GetBestSize())
        pinfo2 = aui.AuiPaneInfo().Name("overlayPanel").Right().Caption('Overlays').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        self._mgr.AddPane(self.overlaypanel, pinfo2)

        self._mgr.AddPane(self.view.CreateToolBar(self), aui.AuiPaneInfo().Name("ViewTools").Caption("View Tools").CloseButton(False).
                          ToolbarPane().Right().GripperTop())


        if self.do.ds.shape[2] > 1:
            self.playbackpanel = playback.PlayPanel(self, self)
            self.playbackpanel.SetSize(self.playbackpanel.GetBestSize())

            pinfo1 = aui.AuiPaneInfo().Name("playbackPanel").Bottom().Caption('Playback').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
            self._mgr.AddPane(self.playbackpanel, pinfo1)

#        self.toolbar = aui.AuiToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,agwStyle=aui.AUI_TB_DEFAULT_STYLE | aui.AUI_TB_OVERFLOW| aui.AUI_TB_VERTICAL)
#        self.toolbar.AddSimpleTool(-1, "Clockwise 1", wx.ArtProvider.GetBitmap(wx.ART_ERROR, wx.ART_OTHER, wx.Size(16, 16)))
#        self.toolbar.Realize()
#        self._mgr.AddPane(self.toolbar, aui.AuiPaneInfo().
#                          Name("toolbar").Caption("Toolbar").
#                          ToolbarPane().Right().GripperTop());

        self._mgr.Update()
        self._mgr.MinimizePane(pinfo)
        self._mgr.MinimizePane(pinfo2)
        #pinfo.Minimize()
        #self._mgr.Update()

        self.updating = False

    def update(self, source=None):
        if not self.updating:
            self.updating = True
            self.view.Refresh()
            if ('update' in dir(self.GetParent())):
                 self.GetParent().update()

            if 'playbackpanel' in dir(self):
                self.playbackpanel.update()

            for cb in self.WantUpdateNotification:
                cb()
            self.updating = False

    def ShowOpts(self, event):
        if (self.showOptsPanel == 1):
            self.showOptsPanel = 0
            self.GetSizer().Show(self.optionspanel, 0)
            self.Layout()
        else:
            self.showOptsPanel = 1
            self.GetSizer().Show(self.optionspanel, 1)
            self.Layout()


        
