#!/usr/bin/python

##################
# viewpanellite.py
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
import sys
sys.path.append(".")

import PYME.cSMI as example

class MyViewPanel(wx.ScrolledWindow):
    def __init__(self, parent, dstack = None):
        wx.ScrolledWindow.__init__(self, parent, -1)

        if (dstack == None):
            self.ds = example.CDataStack("test.kdf")
        else:
            self.ds = dstack

        self.imagepanel = self
        self.selecting = False

        self.SetScrollRate(10, 10)
        
        self.SetVirtualSize(wx.Size(self.ds.getWidth(),self.ds.getHeight()))
        self.SetSize((self.ds.getWidth(),self.ds.getHeight()))

        self.do = example.CDisplayOpts()
        self.do.setDisp1Chan(0)
        self.do.setDisp2Chan(0)
        self.do.setDisp3Chan(0)

        if (self.ds.getNumChannels() >=2):
            self.do.setDisp2Chan(1)
            if (self.ds.getNumChannels() >=3):
                self.do.setDisp1Chan(2)

        self.do.Optimise(self.ds)
        
        self.rend = example.CLUT_RGBRenderer()
        self.rend.setDispOpts(self.do)

        self.scale = 0
        
        self.updating = 0

        self.selection = True

        self.ResetSelection()
        

        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)

        wx.EVT_RIGHT_DOWN(self, self.OnRightDown)
        wx.EVT_RIGHT_UP(self, self.OnRightUp)

        wx.EVT_LEFT_DOWN(self.imagepanel, self.OnRightDown)
        wx.EVT_LEFT_UP(self.imagepanel, self.OnRightUp)

        wx.EVT_MOTION(self.imagepanel, self.OnMotion)
        
        wx.EVT_ERASE_BACKGROUND(self, self.DoNix)

    def SetDataStack(self, ds):
        self.ds = ds
        self.SetVirtualSize(wx.Size(self.ds.getWidth(),self.ds.getHeight()))
        self.SetSize((self.ds.getWidth(),self.ds.getHeight()))

        self.do.setDisp1Chan(0)
        self.do.setDisp2Chan(0)
        self.do.setDisp3Chan(0)

        if (self.ds.getNumChannels() >=2):
            self.do.setDisp2Chan(1)
            if (self.ds.getNumChannels() >=3):
                self.do.setDisp1Chan(2)

        self.do.Optimise(self.ds)

        self.ResetSelection()

        self.Layout()
        self.GetParent().Layout()
        self.Refresh()
        self.Update()



    def DoPaint(self, dc):
        #dc = wx.PaintDC(self.imagepanel)
        #self.imagepanel.PrepareDC(dc)
        #dc.BeginDrawing()
        #mdc = wx.MemoryDC(dc)

        dc.Clear()
        
        s = self.CalcImSize()
        im = wx.EmptyImage(s[0],s[1])
        bmp = im.GetDataBuffer()
        self.rend.pyRender(bmp,self.ds)
        
        sc = pow(2.0,(self.scale))
        im.Rescale(im.GetWidth()*sc,im.GetHeight()*sc) 
        #dc.DrawBitmap(wx.BitmapFromImage(im),wx.Point(0,0))
        dc.DrawBitmap(wx.BitmapFromImage(im),0,0)
        #mdc.SelectObject(wx.BitmapFromImage(self.im))
        #mdc.DrawBitmap(wx.BitmapFromImage(self.im),wx.Point(0,0))
        #dc.Blit(0,0,im.GetWidth(), im.GetHeight(),mdc,0,0)
        #dc.EndDrawing()
            
        if self.selection:
            dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('YELLOW'),0))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)

            if(self.do.getSliceAxis() == self.do.SLICE_XY):
                lx = self.selection_begin_x
                ly = self.selection_begin_y
                hx = self.selection_end_x
                hy = self.selection_end_y
            elif(self.do.getSliceAxis() == self.do.SLICE_XZ):
                lx = self.selection_begin_x
                ly = self.selection_begin_z
                hx = self.selection_end_x
                hy = self.selection_end_z
            elif(self.do.getSliceAxis() == self.do.SLICE_YZ):
                lx = self.selection_begin_y
                ly = self.selection_begin_z
                hx = self.selection_end_y
                hy = self.selection_end_z
        
            #dc.DrawLine((0, ly*sc), (im.GetWidth(), ly*sc))
            #dc.DrawLine((lx*sc, 0), (lx*sc, im.GetHeight()))
            #dc.DrawLine(lx, ly*sc, im.GetWidth(), ly*sc)
            #dc.DrawLine(lx*sc, 0, lx*sc, im.GetHeight())
            
            #(lx*sc,ly*sc, (hx-lx)*sc,(hy-ly)*sc)
            if (self.do.getOrientation() == self.do.UPRIGHT):
                dc.DrawRectangle(lx*sc,ly*sc, (hx-lx)*sc,(hy-ly)*sc)
            else:
                dc.DrawRectangle(ly*sc,lx*sc, (hy-ly)*sc,(hx-lx)*sc)

            dc.SetPen(wx.NullPen)
            dc.SetBrush(wx.NullBrush)

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
            #Perform(WM_ERASEBKGND, MemDC, MemDC);
            #Message.DC := MemDC;
            self.DoPaint(MemDC);
            #Message.DC := 0;
            #DC.BlitXY(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            DC.EndDrawing()
        finally:
            #MemDC.SelectObject(OldBitmap)
            del MemDC
            del MemBitmap

    def OnSize(self,event):
        self.Refresh()
        self.Update()
        
  

    def SetScale(self,scale):
        self.scale = scale

        sc = pow(2.0,(self.scale))
        s = self.CalcImSize()
        self.SetVirtualSize(wx.Size(s[0]*sc,s[1]*sc))
        #self.SetSize(wx.Size(s[0]*sc,s[1]*sc))

        self.Layout()
        self.GetParent().Layout()

        self.Refresh()
        self.Update()

    def Optim(self, event = None):
        self.do.Optimise(self.ds)
        self.updating=1
        self.Refresh()
        self.updating=0

    def CalcImSize(self):
        if (self.do.getSliceAxis() == self.do.SLICE_XY):
            if (self.do.getOrientation() == self.do.UPRIGHT):
                return (self.ds.getWidth(),self.ds.getHeight())
            else:
                return (self.ds.getHeight(),self.ds.getWidth())
        elif (self.do.getSliceAxis() == self.do.SLICE_XZ):
            return (self.ds.getWidth(),self.ds.getDepth())
        else:
            return (self.ds.getHeight(),self.ds.getDepth())

    def DoNix(self, event):
        pass

            
    def OnRightDown(self,event):
        self.selecting = True
        dc = wx.ClientDC(self)
        self.PrepareDC(dc)

        pos = event.GetLogicalPosition(dc)

        print(pos)
        sc = pow(2.0,(self.scale))

        if (self.do.getSliceAxis() == self.do.SLICE_XY):
            self.selection_begin_x = int(pos[0]/sc)
            self.selection_begin_y = int(pos[1]/sc)
        elif (self.do.getSliceAxis() == self.do.SLICE_XZ):
            self.selection_begin_x = int(pos[0]/sc)
            self.selection_begin_z = int(pos[1]/sc)
        elif (self.do.getSliceAxis() == self.do.SLICE_YZ):
            self.selection_begin_y = int(pos[0]/sc)
            self.selection_begin_z = int(pos[1]/sc)

    def OnRightUp(self,event):
        self.selecting = False
        dc = wx.ClientDC(self)
        self.PrepareDC(dc)

        pos = event.GetLogicalPosition(dc)

        print(pos)
        sc = pow(2.0,(self.scale))

        if (self.do.getSliceAxis() == self.do.SLICE_XY):
            self.selection_end_x = int(pos[0]/sc)
            self.selection_end_y = int(pos[1]/sc)
        elif (self.do.getSliceAxis() == self.do.SLICE_XZ):
            self.selection_end_x = int(pos[0]/sc)
            self.selection_end_z = int(pos[1]/sc)
        elif (self.do.getSliceAxis() == self.do.SLICE_YZ):
            self.selection_end_y = int(pos[0]/sc)
            self.selection_end_z = int(pos[1]/sc)

        if ('update' in dir(self.GetParent())):
             self.GetParent().update()
        else:
            self.Refresh()
            self.Update()

    def OnMotion(self, event):
        if event.Dragging() and self.selecting:
            self.ProgressSelection(event)

    def ProgressSelection(self,event):
        dc = wx.ClientDC(self)
        self.PrepareDC(dc)
        pos = event.GetLogicalPosition(dc)
        pos = self.CalcUnscrolledPosition(*pos)
        #print pos
        sc = pow(2.0,(self.scale))
        if (self.do.getSliceAxis() == self.do.SLICE_XY):
            self.selection_end_x = int(pos[0]/sc)
            self.selection_end_y = int(pos[1]/sc)
#        elif (self.do.slice == self.do.SLICE_XZ):
#            self.selection_end_x = int(pos[0]/sc)
#            self.selection_end_z = int(pos[1]/(sc*self.aspect))
#        elif (self.do.slice == self.do.SLICE_YZ):
#            self.selection_end_y = int(pos[0]/sc)
#            self.selection_end_z = int(pos[1]/(sc*self.aspect))
        if ('update' in dir(self.GetParent())):
             self.GetParent().update()
        #self.update()
        else:
            self.Refresh()
            self.Update()
            
    def ResetSelection(self):
        self.selection_begin_x = 0
        self.selection_begin_y = 0
        self.selection_begin_z = 0
        
        self.selection_end_x = self.ds.getWidth() - 1
        self.selection_end_y = self.ds.getHeight() - 1
        self.selection_end_z = self.ds.getDepth() - 1
        
    def SetSelection(self, (b_x,b_y,b_z),(e_x,e_y,e_z)):
        self.selection_begin_x = b_x
        self.selection_begin_y = b_y
        self.selection_begin_z = b_z
        
        self.selection_end_x = e_x
        self.selection_end_y = e_y
        self.selection_end_z = e_z

# end of class ViewPanel
