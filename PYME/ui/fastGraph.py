#!/usr/bin/python

##################
# histLimits.py
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
import wx.lib.newevent
from PYME.ui import wx_compat

import sys,math
import numpy
import os
import numpy as np

#from PYME.Analysis.binAvg import binAvg
from scipy.interpolate import interp1d

LimitChangeEvent, EVT_LIMIT_CHANGE = wx.lib.newevent.NewCommandEvent()

class FastGraphPanel(wx.Panel):
    def __init__(self, parent, id, xvals, data, log=False, size =(200, 100), pos=(0,0), threshMode= False):
        wx.Panel.__init__(self, parent, id, size=size, pos=pos, style=wx.BORDER_SUNKEN)

        self.dragging = None
        self.binSize=None

        self.isinit = False
        

        self.textSize = 10
        self.log = log
        self.left_margin = 20

        self.threshMode = threshMode

        self.SetData(xvals, data)


        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)
        #wx.EVT_LEFT_DOWN(self, self.OnLeftDown)
        #wx.EVT_LEFT_UP(self, self.OnLeftUp)
        #wx.EVT_MOTION(self, self.OnMouseMove)
        #wx.EVT_KEY_DOWN(self, self.OnKeyPress)

        self.isinit = True

    def SetData(self, xvals, data):
        self.data = data.ravel()
        self.xvals = xvals.ravel()

        dSort = numpy.argsort(self.xvals)
        
        #self.data[np.isnan(self.data)]

        #self.upper_pctile = float(data[dSort[len(data)*.99]])
        #self.lower_pctile = float(data[dSort[len(data)*.01]])

        self.xmin = xvals[dSort[0]]
        self.xmax = xvals[dSort[-1]]

        self.limit_lower = float(self.xmin)
        self.limit_upper = float(self.xmax)

        self.GenHist()
        if self.isinit:
            self.Refresh()
            self.Update()

    def GenHist(self):
        #self.hmin = min(self.limit_lower, self.lower_pctile)
        #self.hmax = max(self.limit_upper, self.upper_pctile)
        #if not self.limit_upper >= (self.limit_lower + 1e-9):
        #    self.limit_upper += 1e-9
            
        self.hmin = self.limit_lower
        self.hmax = self.limit_upper
        
        hmid = (self.hmin + self.hmax)/2.0

        #expand shown range to twice current range
        #self.hmin = hmid - 1.5*(hmid-self.hmin)
        #self.hmax = hmid + 1.5*(self.hmax-hmid)

        

        self.hstep = np.float(self.hmax - self.hmin)/max(self.Size[0] - self.left_margin, 1)

        #print self.hmin, self.hmax, self.hstep

        self.edges = numpy.arange(self.hmin, self.hmax, self.hstep)

        #n, m, s = binAvg(self.xvals, self.data, numpy.arange(self.hmin, self.hmax, self.hstep))
        #self.h = m
        
        inter = interp1d(self.xvals, self.data, 'nearest')
        self.h = inter(self.edges)

        if self.log:
            self.h = numpy.log10(self.h+.01) - numpy.log10(.01)

        self.dmean = self.data.mean()
        

    def DoPaint(self, dc):
        if self.Size[0] < 1 or self.Size[1] < 1: #nothing to do
            return
        dc.SetFont(wx.NORMAL_FONT)
        self.textSize = dc.GetTextExtent('test')[1] + 4
        #text_x = dc.GetTextExtent('test')[0]
        

        h = (self.Size[1] - self.textSize - 2)*(1.0-((self.h - self.h.min())/(1.0*(self.h.max()-self.h.min() + .1))))

        maxy = self.Size[1] - self.textSize
        pointlist = [(i+self.left_margin,h_i) for i, h_i in zip(range(len(h)), h)]
        #pointlist = [(10,h[0])] + pointlist + [(self.Size[0], maxy)]

        dc.Clear()

        #when being used to determine histogram bins
        if not self.binSize is None:
            binEdges = numpy.arange(self.hmin, self.hmax + self.binSize, self.binSize)

            for i in range(len(binEdges) -1):
                llx = math.floor((binEdges[i] - self.hmin)/self.hstep)
                ulx = math.ceil((binEdges[i+1] - self.hmin)/self.hstep)

                dc.SetPen(wx.TRANSPARENT_PEN)

                if i % 2 == 0: #even
                    dc.SetBrush(wx.Brush(wx.Colour(0xC0, 0xC0, 0xFF)))
                else:
                    dc.SetBrush(wx.Brush(wx.Colour(0xA0, 0xA0, 0xFF)))

                dc.DrawRectangle(llx, 0, ulx - llx, maxy)

        dc.SetPen(wx.BLACK_PEN)
        dc.SetBrush(wx.BLACK_BRUSH)
        dc.DrawLines(pointlist)
        
        #if self.dragging == 'lower':
        #    dc.SetPen(wx.Pen(wx.GREEN, 2))
        #else:
        #    dc.SetPen(wx.Pen(wx.RED, 2))

        #print self.limit_lower

        llx = (self.limit_lower - self.hmin)/self.hstep
        #dc.DrawLine(llx, 0, llx, maxy)
        lab = '%1.3G' % self.limit_lower
        labSize = dc.GetTextExtent(lab)
        dc.DrawText(lab, max(llx - labSize[0]/2 + self.left_margin, 0), maxy + 2)


        if not self.threshMode:
            #if self.dragging == 'upper':
            #    dc.SetPen(wx.Pen(wx.GREEN, 2))
            #else:
            #    dc.SetPen(wx.Pen(wx.RED, 2))

            ulx = (self.limit_upper - self.hmin)/self.hstep
            #dc.DrawLine(ulx, 0, ulx, maxy)
            lab = '%1.3G' % self.limit_upper
            labSize = dc.GetTextExtent(lab)
            dc.DrawText(lab, min(ulx - labSize[0]/2, self.Size[0] - labSize[0]), maxy + 2)
            
        lly = 0 #self.h.min()
        #dc.DrawLine(llx, 0, llx, maxy)
        lab = '%1.3G' % self.h.max()
        labSize = dc.GetTextExtent(lab)
        dc.DrawText(lab, 0, lly)
        
        # -  #self.h.min()
        #dc.DrawLine(llx, 0, llx, maxy)
        lab = '%1.3G' % self.h.min()
        labSize = dc.GetTextExtent(lab)
        lly = maxy - labSize[1]
        dc.DrawText(lab, 0, lly)


        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
        dc.SetFont(wx.NullFont)

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

            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            #DC.EndDrawing()
        finally:

            del MemDC
            del MemBitmap

    def OnSize(self, event):
        self.GenHist()
        self.Refresh()
        self.Update()

    


class SpecGraphPanel(FastGraphPanel):
    def __init__(self, parent, scope, id=-1):
        FastGraphPanel.__init__(self, parent, id, scope.cam.XVals, scope.cam.XVals)

        self.scope = scope

    def refr(self, sender=None, **kwargs):
        self.SetData(self.scope.cam.XVals, self.scope.frameWrangler.currentFrame)
