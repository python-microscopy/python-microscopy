#!/usr/bin/python

##################
# myviewpanel.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################


import wx
import sys
sys.path.append(".")

import viewpanel
import PYME.cSMI as example

class MyViewPanel(viewpanel.ViewPanel):
    def __init__(self, parent, dstack = None):
        viewpanel.ViewPanel.__init__(self, parent, -1)        

        if (dstack is None):
            self.ds = example.CDataStack("test.kdf")
        else:
            self.ds = dstack

        self.imagepanel.SetVirtualSize(wx.Size(self.ds.getWidth(),self.ds.getHeight()))
        self.imagepanel.SetSize((self.ds.getWidth(),self.ds.getHeight()))

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

        self.cbRedChan.Append("<none>")
        self.cbGreenChan.Append("<none>")
        self.cbBlueChan.Append("<none>")

        for i in range(self.ds.getNumChannels()):
            self.cbRedChan.Append(self.ds.getChannelName(i))
            self.cbGreenChan.Append(self.ds.getChannelName(i))
            self.cbBlueChan.Append(self.ds.getChannelName(i))

        self.scale = 2
        self.crosshairs = True
        self.selection = True
        
        self.ResetSelection()

        self.SetOpts()
        self.updating = 0
        self.showOptsPanel = 1

        wx.EVT_PAINT(self.imagepanel, self.OnPaint)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_KEY_DOWN(self.imagepanel, self.OnKeyPress)
        wx.EVT_LEFT_UP(self.imagepanel, self.OnLeftClick)
        
        wx.EVT_RIGHT_DOWN(self.imagepanel, self.OnRightDown)
        wx.EVT_RIGHT_UP(self.imagepanel, self.OnRightUp)

        wx.EVT_COMBOBOX(self,self.cbRedChan.GetId(), self.GetOpts)
        wx.EVT_COMBOBOX(self,self.cbGreenChan.GetId(), self.GetOpts)
        wx.EVT_COMBOBOX(self,self.cbBlueChan.GetId(), self.GetOpts)
        wx.EVT_COMBOBOX(self,self.cbSlice.GetId(), self.GetOpts)
        wx.EVT_COMBOBOX(self,self.cbScale.GetId(), self.GetOpts)

        wx.EVT_TEXT(self,self.tRedGain.GetId(), self.GetOpts)
        wx.EVT_TEXT(self,self.tRedOff.GetId(), self.GetOpts)
        wx.EVT_TEXT(self,self.tGreenGain.GetId(), self.GetOpts)
        wx.EVT_TEXT(self,self.tGreenOff.GetId(), self.GetOpts)
        wx.EVT_TEXT(self,self.tBlueGain.GetId(), self.GetOpts)
        wx.EVT_TEXT(self,self.tBlueOff.GetId(), self.GetOpts)

        wx.EVT_BUTTON(self, self.bOptimise.GetId(), self.Optim)
        wx.EVT_BUTTON(self, self.bShowOpts.GetId(), self.ShowOpts)

        wx.EVT_ERASE_BACKGROUND(self.imagepanel, self.DoNix)
        wx.EVT_ERASE_BACKGROUND(self, self.DoNix)

    def SetDataStack(self, ds):
        self.ds = ds
        self.imagepanel.SetVirtualSize(wx.Size(self.ds.getWidth(),self.ds.getHeight()))
        self.imagepanel.SetSize((self.ds.getWidth(),self.ds.getHeight()))

        self.do.setDisp1Chan(0)
        self.do.setDisp2Chan(0)
        self.do.setDisp3Chan(0)

        if (self.ds.getNumChannels() >=2):
            self.do.setDisp2Chan(1)
            if (self.ds.getNumChannels() >=3):
                self.do.setDisp1Chan(2)

        self.do.Optimise(self.ds)

        self.cbRedChan.Clear()
        self.cbGreenChan.Clear()
        self.cbBlueChan.Clear()

        self.cbRedChan.Append("<none>")
        self.cbGreenChan.Append("<none>")
        self.cbBlueChan.Append("<none>")

        for i in range(self.ds.getNumChannels()):
            self.cbRedChan.Append(self.ds.getChannelName(i))
            self.cbGreenChan.Append(self.ds.getChannelName(i))
            self.cbBlueChan.Append(self.ds.getChannelName(i))
            
        self.ResetSelection()

        self.SetOpts()
        self.Layout()
        self.Refresh()



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
        
        sc = pow(2.0,(self.scale-2))
        im.Rescale(im.GetWidth()*sc,im.GetHeight()*sc) 
        #dc.DrawBitmap(wx.BitmapFromImage(im),wx.Point(0,0))
        dc.DrawBitmap(wx.BitmapFromImage(im),0,0)
        #mdc.SelectObject(wx.BitmapFromImage(self.im))
        #mdc.DrawBitmap(wx.BitmapFromImage(self.im),wx.Point(0,0))
        #dc.Blit(0,0,im.GetWidth(), im.GetHeight(),mdc,0,0)
        #dc.EndDrawing()

        if self.crosshairs:
            dc.SetPen(wx.Pen(wx.CYAN,0))

            if(self.do.getSliceAxis() == self.do.SLICE_XY):
                lx = self.ds.getXPos()
                ly = self.ds.getYPos()
            elif(self.do.getSliceAxis() == self.do.SLICE_XZ):
                lx = self.ds.getXPos()
                ly = self.ds.getZPos()
            elif(self.do.getSliceAxis() == self.do.SLICE_YZ):
                lx = self.ds.getYPos()
                ly = self.ds.getZPos()
        
            #dc.DrawLine((0, ly*sc), (im.GetWidth(), ly*sc))
            #dc.DrawLine((lx*sc, 0), (lx*sc, im.GetHeight()))
            if (self.do.getOrientation() == self.do.UPRIGHT):
                dc.DrawLine(0, ly*sc, im.GetWidth(), ly*sc)
                dc.DrawLine(lx*sc, 0, lx*sc, im.GetHeight())
            else:
                dc.DrawLine(0, lx*sc, im.GetWidth(), lx*sc)
                dc.DrawLine(ly*sc, 0, ly*sc, im.GetHeight())

            dc.SetPen(wx.NullPen)
            
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
        DC = wx.PaintDC(self.imagepanel)
        self.imagepanel.PrepareDC(DC)
        s = self.imagepanel.GetVirtualSize()
        MemBitmap = wx.EmptyBitmap(s.GetWidth(), s.GetHeight())
        #del DC
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)
        try:
            #DC.BeginDrawing()
            #Perform(WM_ERASEBKGND, MemDC, MemDC);
            #Message.DC := MemDC;
            self.DoPaint(MemDC);
            #Message.DC := 0;
            #DC.BlitXY(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            #DC.EndDrawing()
        finally:
            #MemDC.SelectObject(OldBitmap)
            del MemDC
            del MemBitmap


    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        if rot < 0:
            self.ds.setZPos(self.ds.getZPos() - 1)

        if rot > 0:
            self.ds.setZPos(self.ds.getZPos() + 1)

        if ('update' in dir(self.GetParent())):
             self.GetParent().update()
        else:
            self.imagepanel.Refresh()
    
    def OnKeyPress(self, event):
        if event.GetKeyCode() == wx.WXK_PRIOR:
            self.ds.setZPos(self.ds.getZPos() - 1)
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == wx.WXK_NEXT:
            self.ds.setZPos(self.ds.getZPos() + 1)
            if ('update' in dir(self.GetParent())):
                self.GetParent().update()
            else:
                self.imagepanel.Refresh()
        elif event.GetKeyCode() == 74:
                self.ds.setXPos(self.ds.getXPos() - 1)
                if ('update' in dir(self.GetParent())):
                    self.GetParent().update()
                else:
                    self.imagepanel.Refresh()
        elif event.GetKeyCode() == 76:
                self.ds.setXPos(self.ds.getXPos() + 1)
                if ('update' in dir(self.GetParent())):
                    self.GetParent().update()
                else:
                    self.imagepanel.Refresh()
        elif event.GetKeyCode() == 79:
                self.ds.setYPos(self.ds.getYPos() + 1)
                if ('update' in dir(self.GetParent())):
                    self.GetParent().update()
                else:
                    self.imagepanel.Refresh()
        elif event.GetKeyCode() == 77:
                self.ds.setYPos(self.ds.getYPos() - 1)
                if ('update' in dir(self.GetParent())):
                    self.GetParent().update()
                else:
                    self.imagepanel.Refresh()
        else:
            event.Skip()
        
    def SetOpts(self,event=None):
        self.cbRedChan.SetSelection(self.do.getDisp3Chan() + 1)
        self.cbGreenChan.SetSelection(self.do.getDisp2Chan() + 1)
        self.cbBlueChan.SetSelection(self.do.getDisp1Chan() + 1)

        self.tRedGain.SetValue(self.do.getDisp3Gain().__str__())
        self.tGreenGain.SetValue(self.do.getDisp2Gain().__str__())
        self.tBlueGain.SetValue(self.do.getDisp1Gain().__str__())

        self.tRedOff.SetValue(self.do.getDisp3Off().__str__())
        self.tGreenOff.SetValue(self.do.getDisp2Off().__str__())
        self.tBlueOff.SetValue(self.do.getDisp1Off().__str__())

        if (self.do.getSliceAxis() == self.do.SLICE_XY):
            if (self.do.getOrientation() == self.do.UPRIGHT):
                self.cbSlice.SetSelection(0)
            else:
                self.cbSlice.SetSelection(1)
        elif (self.do.getSliceAxis() == self.do.SLICE_XZ):
            self.cbSlice.SetSelection(2)
        else:
            self.cbSlice.SetSelection(3)

        self.cbScale.SetSelection(self.scale)

    def GetOpts(self,event=None):
        if (self.updating == 0):
            self.do.setDisp3Chan(self.cbRedChan.GetSelection() - 1)
            self.do.setDisp2Chan(self.cbGreenChan.GetSelection() - 1)
            self.do.setDisp1Chan(self.cbBlueChan.GetSelection() - 1)

            self.do.setDisp3Gain(float(self.tRedGain.GetValue()))
            self.do.setDisp2Gain(float(self.tGreenGain.GetValue()))
            self.do.setDisp1Gain(float(self.tBlueGain.GetValue()))

            self.do.setDisp3Off(int(self.tRedOff.GetValue()))
            self.do.setDisp2Off(int(self.tGreenOff.GetValue()))
            self.do.setDisp1Off(int(self.tBlueOff.GetValue()))

            if (self.cbSlice.GetSelection() == 0):
                self.do.setSliceAxis(self.do.SLICE_XY)
                self.do.setOrientation(self.do.UPRIGHT)
            elif (self.cbSlice.GetSelection() == 1):
                self.do.setSliceAxis(self.do.SLICE_XY)
                self.do.setOrientation(self.do.ROT90)
            elif (self.cbSlice.GetSelection() == 2):
                self.do.setSliceAxis(self.do.SLICE_XZ)
                self.do.setOrientation(self.do.UPRIGHT)
            elif (self.cbSlice.GetSelection() == 3):
                self.do.setSliceAxis(self.do.SLICE_YZ)
                self.do.setOrientation(self.do.UPRIGHT)   

            self.scale = self.cbScale.GetSelection()

            sc = pow(2.0,(self.scale-2))
            s = self.CalcImSize()
            self.imagepanel.SetVirtualSize(wx.Size(s[0]*sc,s[1]*sc))

            self.imagepanel.Refresh()

    def Optim(self, event = None):
        self.do.Optimise(self.ds)
        self.updating=1
        self.SetOpts()
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

    def ShowOpts(self, event):
        if (self.showOptsPanel == 1):
            self.showOptsPanel = 0
            self.GetSizer().Show(self.optionspanel, 0)
            self.Layout()
        else:
            self.showOptsPanel = 1
            self.GetSizer().Show(self.optionspanel, 1)
            self.Layout()

    def OnLeftClick(self,event):
        dc = wx.ClientDC(self.imagepanel)
        self.imagepanel.PrepareDC(dc)

        pos = event.GetLogicalPosition(dc)

        #print pos
        sc = pow(2.0,(self.scale-2))

        if (self.do.getSliceAxis() == self.do.SLICE_XY):
            self.ds.setXPos(int(pos[0]/sc))
            self.ds.setYPos(int(pos[1]/sc))
        elif (self.do.getSliceAxis() == self.do.SLICE_XZ):
            self.ds.setXPos(int(pos[0]/sc))
            self.ds.setZPos(int(pos[1]/sc))
        elif (self.do.getSliceAxis() == self.do.SLICE_YZ):
            self.ds.setYPos(int(pos[0]/sc))
            self.ds.setZPos(int(pos[1]/sc))

        if ('update' in dir(self.GetParent())):
             self.GetParent().update()
        else:
            self.imagepanel.Refresh()
            
    def OnRightDown(self,event):
        dc = wx.ClientDC(self.imagepanel)
        self.imagepanel.PrepareDC(dc)

        pos = event.GetLogicalPosition(dc)

        print(pos)
        sc = pow(2.0,(self.scale-2))

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
        dc = wx.ClientDC(self.imagepanel)
        self.imagepanel.PrepareDC(dc)

        pos = event.GetLogicalPosition(dc)

        print(pos)
        sc = pow(2.0,(self.scale-2))

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
            self.imagepanel.Refresh()
            
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
