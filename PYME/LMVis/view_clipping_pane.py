# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:20:11 2016

@author: david
"""
import wx
import wx.lib.newevent
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import numpy as np

LimitChangeEvent, EVT_LIMIT_CHANGE = wx.lib.newevent.NewCommandEvent()

import sys
if sys.platform == 'darwin':
    # osx gives us LOTS of scroll events
    # ajust the mag in smaller increments
    SCROLL_FACTOR = .02
else:
    SCROLL_FACTOR = .02



from PYME.recipes.traits import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, on_trait_change


class PointDisplaySettings(HasTraits):
    pointSize = Float(5.0)
    colourDataKey = CStr('t')
    alpha = Float(1.0)


def _getPossibleKeys(pipeline):
    colKeys = ['<None>']
    
    if not pipeline.colourFilter is None: #is the test needed?
        colKeys += list(pipeline.keys())
    
    colKeys += list(pipeline.GeneratedMeasures.keys())
    
    colKeys.sort()
    
    return colKeys

class ClippingPanel(wx.Panel):
    def __init__(self, parent, id, glcanvas, axis='x', log=False, size=(300, 30), pos=(0, 0),
                 threshMode=False):
        wx.Panel.__init__(self, parent, id, size=size, pos=pos)#, style=wx.BORDER_SUNKEN)
        
        self.glcanvas = glcanvas
        self.axis = axis
        
        self.dragging = None   # are we updating view_limits with a mouse drag?
        self.editing = None  # are we updating view_limits with a text box?

        
        #self.data_limits = data_limits
        #self.view_limits = view_lim
        
        #print 'r', self.view_limits, self.view_limits[0]
        
        self.view_limits[0] = max(self.view_limits[0], self.data_limits[0])
        self.view_limits[1] = min(self.view_limits[1], self.data_limits[1])

        #print 'm', self.view_limits, self.view_limits[0]
        
        self.textSize = 10

        # Text controls for upper and lower bounds
        text_ctrl_width = 90
        self.ll_ctrl = wx.TextCtrl(self, -1, 
                                   str(self.view_limits[0]), 
                                   size=(text_ctrl_width, -1), 
                                   name='lower_limit', 
                                   style=wx.TE_PROCESS_ENTER)
        self.ul_ctrl = wx.TextCtrl(self, -1, 
                                   str(self.view_limits[1]), 
                                   size=(text_ctrl_width, -1), 
                                   name='upper_limit', 
                                   style=wx.TE_PROCESS_ENTER)
        maxy = self.Size[1] - self.ll_ctrl.Size[1] + 2
        self.ll_ctrl.SetPosition((0,maxy))
        self.ul_ctrl.SetPosition((self.Size[0]-self.ul_ctrl.Size[0],maxy))
        self.ll_ctrl.Hide()
        self.ul_ctrl.Hide()
        self.ll_ctrl.Bind(wx.EVT_KILL_FOCUS, self.HideTextCtrl)
        self.ul_ctrl.Bind(wx.EVT_KILL_FOCUS, self.HideTextCtrl)
        self.ll_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnTextCtrlEnter)
        self.ul_ctrl.Bind(wx.EVT_TEXT_ENTER, self.OnTextCtrlEnter)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        #wx.EVT_KEY_DOWN(self, self.OnKeyPress)
        #wx.EVT_RIGHT_UP(self, self.OnRightUp)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseScrollEvent)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
    
    @property
    def data_limits(self):
        bb = self.glcanvas.bbox
        
        #print 'bbox: ' + repr(bb)
        
        if bb is None:
            return [-1e6, 1e6]
        
        
        if self.axis == 'x':
            return bb[0]-1, bb[3]+1
        elif self.axis == 'y':
            return bb[1]-1, bb[4]+1
        elif self.axis == 'z':
            return bb[2]-1, bb[5]+1
        else:
            return 0, 1
        
    @property
    def view_limits(self):
        #print 'q', self.glcanvas.bounds
        #print self.glcanvas.bounds[self.axis]
        vl = self.glcanvas.bounds[self.axis][0]
        #print vl
        return vl
        
        
        #return max(vl[0], self.data_limits[0]), min(vl[1], self.data_limits[1])
        
    def _get_coords(self):
        wp = self.Size[0] - 10
        x0p = 5
    
        x0, xmx = self.data_limits
        self.view_limits[0] = max(self.view_limits[0], x0)
        self.view_limits[1] = min(self.view_limits[1], xmx)
    
        w = xmx - x0
    
        x0v, xmxv = self.view_limits
    
        x0vp = x0p + wp * float(x0v - x0) / w
        xmvp = x0p + wp * float(xmxv - x0) / w
        
        return x0p, wp, x0vp, xmvp, w, x0
        

    def DoPaint(self, dc):
        
        
        if self.Size[0] < 1 or self.Size[1] < 1: #nothing to do
            return
        dc.SetFont(wx.NORMAL_FONT)
        self.textSize = dc.GetTextExtent('test')[1] + 4
    
        #h = (self.Size[1] - self.textSize - 2) * (1.0 - (self.h / (1.0 * self.h[1:-1].max() + 1e-9))) + 2
    
        maxy = self.Size[1] - self.textSize
    
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        
        x0p, wp, x0vp, xmvp, w, x0 = self._get_coords()
            
        dc.DrawRectangle(x0p, 5, wp, maxy-5)
    
        dc.SetPen(wx.BLACK_PEN)
        dc.SetBrush(wx.BLACK_BRUSH)

        dc.DrawRectangle(x0vp, 5, xmvp - x0vp, maxy-5)

        #draw lines
        #lower
        if self.dragging == 'lower':
            dc.SetPen(wx.Pen(wx.GREEN, 2))
        else:
            dc.SetPen(wx.Pen(wx.RED, 2))
        
        dc.DrawLine(x0vp, 5, x0vp, maxy)
        lab = '%d' % self.view_limits[0]
        labSize = dc.GetTextExtent(lab)
        dc.DrawText(lab, max(x0vp - labSize[0] / 2, 0), maxy + 2)
    
        
        #upper
        if self.dragging == 'upper':
            dc.SetPen(wx.Pen(wx.GREEN, 2))
        else:
            dc.SetPen(wx.Pen(wx.RED, 2))
    
        dc.DrawLine(xmvp, 5, xmvp, maxy)
        lab = '%d' % self.view_limits[1]
        labSize = dc.GetTextExtent(lab)
        dc.DrawText(lab, min(xmvp - labSize[0] / 2, self.Size[0] - labSize[0]), maxy + 2)
    
        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
        dc.SetFont(wx.NullFont)

    def OnPaint(self, event):
        DC = wx.PaintDC(self)
        #self.PrepareDC(DC)
    
        s = self.GetVirtualSize()
        MemBitmap = wx.EmptyBitmap(s.GetWidth(), s.GetHeight())
        #del DC
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)
        try:
            #DC.BeginDrawing()
        
            self.DoPaint(MemDC)
        
            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            #DC.EndDrawing()
        finally:
        
            del MemDC
            del MemBitmap

    def OnSize(self, event):
        self.Refresh()
        self.Update()

    def OnMouseScrollEvent(self, evt):
        rot = evt.GetWheelRotation()
        # shift_offset = self.hstep
        shift_offset = (self.view_limits[1] - self.view_limits[0]) * SCROLL_FACTOR
        # if rot > 0:
        #     delta = shift_offset
        # else:
        #     delta = -shift_offset
        delta = max(min(rot*shift_offset,  self.data_limits[1] - self.view_limits[1]), self.data_limits[0] - self.view_limits[0])
        self.view_limits[0] += delta
        self.view_limits[1] += delta
        
        #print('clip scroll - delta = %g, rot = %g' % (delta, rot))
        
        self.Refresh()
        self.Update()
        self.glcanvas.refresh()
        evt = LimitChangeEvent(self.GetId(), upper=self.view_limits[1], lower=self.view_limits[0])
        #self.ProcessEvent(evt)


    def OnLeftDown(self, event):
        x = event.GetX()
        y = event.GetY()

        x0p, wp, x0vp, xmvp, w, xo = self._get_coords()
        maxy = self.Size[1] - self.ll_ctrl.Size[1]/2 + 2
    
        #hit test the limits
        # check for clipping update via text control
        # Look for positioning near the upper or lower limit markers, but below
        # the draggable bar, with a hittest tolerance the size of the text control box
        ll, ul = max(x0vp-self.ll_ctrl.Size[0]/2, 0), min(xmvp-self.ul_ctrl.Size[0]/2, self.Size[0]-self.ul_ctrl.Size[0])
        lu, uu = ll+self.ll_ctrl.Size[0], ul+self.ul_ctrl.Size[0]
        if (abs(y-maxy) < self.ll_ctrl.Size[1]/2) and (x > ll) and (x < lu):
            self.editing = 'lower'
        if (abs(y-maxy) < self.ul_ctrl.Size[1]/2) and (x > ul) and (x < uu):
            self.editing = 'upper'

        # check for clipping update via dragging
        HITTEST_TOLERANCE = 15
        if abs(x0vp - x) < HITTEST_TOLERANCE:
            self.dragging = 'lower'
        elif abs(xmvp - x) < HITTEST_TOLERANCE:
            self.dragging = 'upper'
        elif x0vp < x < xmvp:
            self.dragging = 'shift'
    
        event.Skip()

    def OnLeftUp(self, event):
        #x = event.GetX()
        #y = event.GetY()
    
        if not self.editing is None:
            x0p, wp, x0vp, xmvp, w, xo = self._get_coords()
            if self.editing == 'lower':
                ll = max(x0vp-self.ll_ctrl.Size[0]/2, 0)
                self.ShowTextCtrl(self.ll_ctrl, self.view_limits[0], ll)
            elif self.editing == 'upper':
                ul = min(xmvp-self.ul_ctrl.Size[0]/2, self.Size[0]-self.ul_ctrl.Size[0])
                self.ShowTextCtrl(self.ul_ctrl, self.view_limits[1], ul)
        elif not self.dragging is None:
            evt = LimitChangeEvent(self.GetId(), upper=self.view_limits[1], lower=self.view_limits[0])
            #evt.ShouldPropagate()
            #wx.PostEvent(self, evt)
            self.ProcessEvent(evt)
            self.glcanvas.refresh()
                
        self.dragging = None
        self.editing = None
        
        self.Refresh()
        self.Update()
        event.Skip()
        
    def OnDoubleClick(self, event):
        dlg = wx.TextEntryDialog(self, 'Clipping range [nm]', 'Restrict clipping to a given range', '200')
        if (dlg.ShowModal() == wx.ID_OK):
            clip_size = float(dlg.GetValue())
            
            vc = 0.5*(self.view_limits[0] + self.view_limits[1])
            
            self.view_limits[0] = vc - 0.5*clip_size
            self.view_limits[1] = vc + 0.5*clip_size

            self.Refresh()
            self.Update()
            self.glcanvas.refresh()
            
            
        dlg.Destroy()


    def OnMouseMove(self, event):
        x = event.GetX()
        #y = event.GetY()

        x0p, wp, x0vp, xmvp, w, x0 = self._get_coords()
    
        #xt = self.hmin + x * self.hstep
        xt = w*float(x - x0p)/wp + x0
    

        if self.dragging == 'lower' and not xt >= self.view_limits[1]:
            self.view_limits[0] = xt
        elif self.dragging == 'upper' and not xt <= self.view_limits[0]:
            self.view_limits[1] = xt
        elif self.dragging == 'shift':
            width = self.view_limits[1] - self.view_limits[0]
            self.view_limits[0] = xt - width / 2
            self.view_limits[1] = xt + width / 2
            
        if self.dragging:
            self.Refresh()
            self.Update()
    
        event.Skip()

    def ShowTextCtrl(self, text_ctrl, value, xpos):
        # Update text box position and value
        ypos = self.Size[1] - text_ctrl.Size[1] + 2
        text_ctrl.SetPosition((xpos, ypos))
        text_ctrl.SetValue(str(value))

        # Display
        text_ctrl.Show()
        text_ctrl.SetFocus()

    def HideTextCtrl(self, event):
        event.GetEventObject().Hide()
        event.Skip()

    def OnTextCtrlEnter(self, event):
        text_ctrl = event.GetEventObject()
        if text_ctrl.GetId() == self.ll_ctrl.GetId():
            self.view_limits[0] = float(text_ctrl.GetValue())
        elif text_ctrl.GetId() == self.ul_ctrl.GetId():
            self.view_limits[1] = float(text_ctrl.GetValue())
        text_ctrl.Hide()
        self.Refresh()
        self.Update()
        self.glcanvas.refresh()

        # event.Skip()  # purposefully commented out to prevent conflict with wx.EVT_KILL_FOCUS


class ViewClippingPanel(wx.Panel):
    """A GUI class for dynamically filtering points 
    spatially by adjusting sliders.
    """
    
    def __init__(self, parent, glcanvas):
        wx.Panel.__init__(self, parent, -1)
        
        self.glcanvas = glcanvas
        #self.pointDisplaySettings = pointDisplaySettings
        
        bsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'x:'), 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)
        
        self.sx = ClippingPanel(self, -1, self.glcanvas, 'x')
        hsizer.Add(self.sx, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        
        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'y:'), 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.sy = ClippingPanel(self, -1, self.glcanvas, 'y')
        hsizer.Add(self.sy, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'z:'), 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.sz = ClippingPanel(self, -1, self.glcanvas, 'z')
        hsizer.Add(self.sz, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)
        
        self.SetSizerAndFit(bsizer)
        
        
    



def GenViewClippingPanel(visgui, pnl, title='View Clipping'):
    """ 
    Generate a folding pane that lets a user dynamically filter points 
    spatially by adjusting sliders.
    """
    item = afp.foldingPane(pnl, -1, caption=title, pinned=False)
    
    pan = ViewClippingPanel(item, visgui.glCanvas)
    item.AddNewElement(pan)
    pnl.AddPane(item)