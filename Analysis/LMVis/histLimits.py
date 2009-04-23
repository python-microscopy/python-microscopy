import wx
import wx.lib.newevent

import sys,math
import numpy
import os

LimitChangeEvent, EVT_LIMIT_CHANGE = wx.lib.newevent.NewCommandEvent()

class HistLimitPanel(wx.Panel):
    def __init__(self, parent, id, data, limit_lower, limit_upper, log=False, size =(200, 100), pos=(0,0)):
        wx.Panel.__init__(self, parent, id, size=size, pos=pos, style=wx.BORDER_SUNKEN)
        
        self.data = data.ravel()

        self.dragging = None
        
        dSort = numpy.argsort(data)
        
        self.upper_pctile = float(data[dSort[len(data)*.95]])
        self.lower_pctile = float(data[dSort[len(data)*.05]])

        self.limit_lower = limit_lower
        self.limit_upper = limit_upper

        self.textSize = 10
        self.log = log

        self.GenHist()


        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_LEFT_DOWN(self, self.OnLeftDown)
        wx.EVT_LEFT_UP(self, self.OnLeftUp)
        wx.EVT_MOTION(self, self.OnMouseMove)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)

    def SetData(self, data, lower, upper):
        self.data = data.ravel()

        dSort = numpy.argsort(data)

        self.upper_pctile = float(data[dSort[len(data)*.95]])
        self.lower_pctile = float(data[dSort[len(data)*.05]])

        self.limit_lower = lower
        self.limit_upper = upper

        self.GenHist()
        self.Refresh()

    def OnLeftDown(self, event):
        x = event.GetX()
        #y = event.GetY()

        #hit test the limits
        llx = (self.limit_lower - self.hmin)/self.hstep
        ulx = (self.limit_upper - self.hmin)/self.hstep
        if abs(llx - x) < 3:
            self.dragging = 'lower'
        elif abs(ulx - x) < 3:
            self.dragging = 'upper'

        event.Skip()

    def OnLeftUp(self, event):
        #x = event.GetX()
        #y = event.GetY()

        if not self.dragging == None:
            evt = LimitChangeEvent(self.GetId(), upper=self.limit_upper, lower=self.limit_lower)
            wx.PostEvent(self, evt)

        self.dragging=None
        self.GenHist()
        self.Refresh()
        event.Skip()

    def OnMouseMove(self, event):
        x = event.GetX()
        #y = event.GetY()
        
        xt = self.hmin + x*self.hstep

        if self.dragging == 'lower' and not xt >= self.limit_upper:
            self.limit_lower = xt
        elif self.dragging == 'upper' and not xt <= self.limit_lower:
            self.limit_upper = xt

        self.Refresh()

        event.Skip()

    def GenHist(self):
        #self.hmin = min(self.limit_lower, self.lower_pctile)
        #self.hmax = max(self.limit_upper, self.upper_pctile)

        self.hmin = self.limit_lower
        self.hmax = self.limit_upper
        hmid = (self.hmin + self.hmax)/2.0

        #expand shown range to twice current range
        self.hmin = hmid - 1.5*(hmid-self.hmin)
        self.hmax = hmid + 1.5*(self.hmax-hmid)

        self.hstep = (self.hmax - self.hmin)/self.Size[0]

        self.h, self.edges = numpy.histogram(self.data, numpy.arange(self.hmin, self.hmax, self.hstep))

        if self.log:
            self.h = numpy.log10(self.h+.01) - numpy.log10(.01)

    def DoPaint(self, dc):
        dc.SetFont(wx.NORMAL_FONT)
        self.textSize = dc.GetTextExtent('test')[1] + 4

        h = (self.Size[1] - self.textSize - 2)*(1.0-(self.h/(1.0*self.h[1:-1].max()))) + 2

        maxy = self.Size[1] - self.textSize
        pointlist = [(i,h_i) for i, h_i in zip(range(len(h)), h)]
        pointlist = [(0,maxy)] + pointlist + [(self.Size[0], maxy)]

        dc.Clear()

        dc.SetPen(wx.BLACK_PEN)
        dc.SetBrush(wx.BLACK_BRUSH)
        dc.DrawPolygon(pointlist)
        
        if self.dragging == 'lower':
            dc.SetPen(wx.Pen(wx.GREEN, 2))
        else:
            dc.SetPen(wx.Pen(wx.RED, 2))

        llx = (self.limit_lower - self.hmin)/self.hstep
        dc.DrawLine(llx, 0, llx, maxy)
        lab = '%1.3G' % self.limit_lower
        labSize = dc.GetTextExtent(lab)
        dc.DrawText(lab, max(llx - labSize[0]/2, 0), maxy + 2)

        if self.dragging == 'upper':
            dc.SetPen(wx.Pen(wx.GREEN, 2))
        else:
            dc.SetPen(wx.Pen(wx.RED, 2))
            
        ulx = (self.limit_upper - self.hmin)/self.hstep
        dc.DrawLine(ulx, 0, ulx, maxy)
        lab = '%1.3G' % self.limit_upper
        labSize = dc.GetTextExtent(lab)
        dc.DrawText(lab, min(ulx - labSize[0]/2, self.Size[0] - labSize[0]), maxy + 2)

        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
        dc.SetFont(wx.NullFont)

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

    def OnKeyPress(self, event):
        if event.GetKeyCode() == 76: #L - toggle log y axis
            self.log = not self.log
            self.GenHist()
            self.Refresh()
        elif event.GetKeyCode() == 77: #M - set min-max
            self.limit_lower = self.data.min()
            self.limit_upper = self.data.max()
            self.GenHist()
            self.Refresh()
            evt = LimitChangeEvent(self.GetId(), upper=self.limit_upper, lower=self.limit_lower)
            wx.PostEvent(self, evt)
        elif event.GetKeyCode() == 80: #P - set percentile
            self.limit_lower = self.lower_pctile
            self.limit_upper = self.upper_pctile
            self.GenHist()
            self.Refresh()
            evt = LimitChangeEvent(self.GetId(), upper=self.limit_upper, lower=self.limit_lower)
            wx.PostEvent(self, evt)
        else:
            event.Skip()

    def SetValue(self, val):
        self.limit_lower = val[0]
        self.limit_upper = val[1]

        self.GenHist()
        self.Refresh()

    def GetValue(self):
        return (self.limit_lower, self.limit_upper)


def ShowHistLimitFrame(parent, title, data, limit_lower, limit_upper, size=(200, 100), log=False):
    f = wx.Frame(parent, title=title, size=size)
    ID_HIST_LIM = wx.NewId()
    p = HistLimitPanel(f,ID_HIST_LIM, data, limit_lower, limit_upper, log=log)
    f.Show()
    return ID_HIST_LIM

class HistLimitDialog(wx.Dialog):
    def __init__(self, parent, data,lower, upper, title=''):
        wx.Dialog.__init__(self, parent, title=title)

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.hl = HistLimitPanel(self, -1, data, lower, upper, size=(200, 100))
        sizer1.Add(self.hl, 0, wx.ALL|wx.EXPAND, 5)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        #btn = wx.Button(self, wx.ID_CANCEL)

        #btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def GetLimits(self):
        return self.hl.GetValue()