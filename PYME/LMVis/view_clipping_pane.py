# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:20:11 2016

@author: david
"""
import wx
import wx.lib.newevent
import PYME.ui.autoFoldPanel as afp
import numpy as np

LimitChangeEvent, EVT_LIMIT_CHANGE = wx.lib.newevent.NewCommandEvent()

import sys
if sys.platform == 'darwin':
    # osx gives us LOTS of scroll events
    # ajust the mag in smaller increments
    SCROLL_FACTOR = .02
else:
    SCROLL_FACTOR = .2


try:
    from enthought.traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, \
        ListInstance, on_trait_change
    #from enthought.traits.ui.api import View, Item, EnumEditor, InstanceEditor, Group
except ImportError:
    from traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, ListInstance, \
        on_trait_change
    #from traitsui.api import View, Item, EnumEditor, InstanceEditor, Group


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
        
        self.dragging = None

        
        #self.data_limits = data_limits
        #self.view_limits = view_lim
        
        self.view_limits[0] = max(self.view_limits[0], self.data_limits[0])
        self.view_limits[1] = min(self.view_limits[1], self.data_limits[1])
        
        self.textSize = 10
        
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_LEFT_DOWN(self, self.OnLeftDown)
        wx.EVT_LEFT_UP(self, self.OnLeftUp)
        wx.EVT_MOTION(self, self.OnMouseMove)
        #wx.EVT_KEY_DOWN(self, self.OnKeyPress)
        #wx.EVT_RIGHT_UP(self, self.OnRightUp)
        wx.EVT_MOUSEWHEEL(self, self.OnMouseScrollEvent)
    
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
        return self.glcanvas.bounds[self.axis]
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
    
        dc.SetBackground(wx.TRANSPARENT_BRUSH)
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
            DC.BeginDrawing()
        
            self.DoPaint(MemDC)
        
            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            DC.EndDrawing()
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
        #y = event.GetY()

        x0p, wp, x0vp, xmvp, w, xo = self._get_coords()
    
        #hit test the limits
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
    
        if not self.dragging is None:
            evt = LimitChangeEvent(self.GetId(), upper=self.view_limits[1], lower=self.view_limits[0])
            #evt.ShouldPropagate()
            #wx.PostEvent(self, evt)
            self.ProcessEvent(evt)
            self.glcanvas.refresh()
    
        self.dragging = None
        
        self.Refresh()
        self.Update()
        event.Skip()


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


class ViewClippingPanel(wx.Panel):
    """A GUI class for determining the settings to use when displaying points
    in VisGUI.

    Constructed as follows:
    PointSettingsPanel(parent, pipeline, pointDisplaySettings)

    where:
      parent is the parent window
      pipeline is the pipeline object which provides the points,
      pointDisplaySettings is an instance of PointDisplaySettings


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
        
        
    



def GenViewClippingPanel(visgui, pnl, title='Clipping'):
    """Generate a ponts pane and insert into the given panel"""
    item = afp.foldingPane(pnl, -1, caption=title, pinned=True)
    
    pan = ViewClippingPanel(item, visgui.glCanvas)
    item.AddNewElement(pan)
    pnl.AddPane(item)