#!/usr/bin/python
##################
# psfExtraction.py
#
# Copyright David Baddeley, 2011
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
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import numpy as np

from ._base import Plugin

import logging
logger = logging.getLogger(__name__)

class ObjectTagger(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        try:  # stack multiview channels
            self.numChan = self.image.mdh['Multiview.NumROIs']
        except:
            self.ChanOffsetZnm = None
            self.numChan = self.image.data.shape[3]

        self.positions = []
        self.display_size = [30,30,30]

        self._array_view = None
        
        dsviewer.view.add_overlay(self.DrawOverlays, 'Tagged objects')

        dsviewer.paneHooks.append(self.GenPSFPanel)

    def GenPSFPanel(self, _pnl):
        item = afp.foldingPane(_pnl, -1, caption="Object Tagging", pinned = True)

        pan = wx.Panel(item, -1)

        vsizer = wx.BoxSizer(wx.VERTICAL)            
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        hsizer.Add(wx.StaticText(pan, -1, 'Threshold:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
#        self.tThreshold = wx.TextCtrl(pan, -1, value='50', size=(40, -1))
#
        bTagPSF = wx.Button(pan, -1, 'Tag', style=wx.BU_EXACTFIT)
        bTagPSF.Bind(wx.EVT_BUTTON, self.OnTagObject)
        hsizer.Add(bTagPSF, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bClearTagged = wx.Button(pan, -1, 'Clear', style=wx.BU_EXACTFIT)
        bClearTagged.Bind(wx.EVT_BUTTON, self.OnClearTags)
        hsizer.Add(bClearTagged, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0,wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(pan, -1, 'Display Half Size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPSFROI = wx.TextCtrl(pan, -1, value='30,30,30', size=(40, -1))
        hsizer.Add(self.tPSFROI, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tPSFROI.Bind(wx.EVT_TEXT, self.OnPSFROI)

        vsizer.Add(hsizer, 0,wx.EXPAND|wx.ALL, 0)

        pan.SetSizer(vsizer)
        vsizer.Fit(pan)

        item.AddNewElement(pan)
        _pnl.AddPane(item)



    def OnTagObject(self, event):
        from PYME.Analysis.PSFEst import extractImages
        #if we already have a location there, un-tag it
        for i, p in enumerate(self.positions):
            if ((np.array(p[:2]) - np.array((self.do.xp, self.do.yp)))**2).sum() < 100:
                self.positions.pop(i)
                self._update_array_view()
                self.view.Refresh()

                return
                
        self.positions.append((self.do.xp, self.do.yp, self.do.zp))
        #self.view.psfROIs = self.PSFLocs
        
        self._update_array_view()
        self.view.Refresh()

    def _update_array_view(self):
        from PYME.ui import recArrayView

        data = np.array(self.positions, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        if self._array_view is None:
            self._array_view = recArrayView.ArrayPanel(self.dsviewer, data)
            self.dsviewer.AddPage(self._array_view, caption='Object Positions')
        else:
            self._array_view.grid.SetData(data)


    def OnClearTags(self, event):
        self.positions = []
        #self.view.psfROIs = self.PSFLocs
        self.view.Refresh()

    def OnPSFROI(self, event):
        try:
            self.display_size = [int(s) for s in self.tPSFROI.GetValue().split(',')]
            #self.view.psfROISize = psfROISize
            self.view.Refresh()
        except:
            pass
        
    def DrawOverlays(self, view, dc):
        #PSF ROIs
        if (len(self.positions) > 0):
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.Pen(wx.TheColourDatabase.FindColour('GREEN'),1))
            
            if(view.do.slice == view.do.SLICE_XY):
                a_x = 0
                a_y = 1
            elif(view.do.slice == view.do.SLICE_XZ):
                a_x = 0
                a_y = 2
            elif(view.do.slice == view.do.SLICE_YZ):
                a_x = 1
                a_y = 2
                
            for p in self.positions:
                #dc.DrawRectangle(sc*p[0]-self.psfROISize[0]*sc - x0,sc*p[1] - self.psfROISize[1]*sc - y0, 2*self.psfROISize[0]*sc,2*self.psfROISize[1]*sc)
                xp0, yp0 = view.pixel_to_screen_coordinates(p[a_x]-self.display_size[a_x],p[a_y] - self.display_size[a_y])
                xp1, yp1 = view.pixel_to_screen_coordinates(p[a_x]+self.display_size[a_x],p[a_y] + self.display_size[a_y])
                dc.DrawRectangle(xp0, yp0, xp1-xp0,yp1-yp0)



def Plug(dsviewer):
    return ObjectTagger(dsviewer)

