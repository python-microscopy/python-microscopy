#!/usr/bin/python

##################
# DisplayOptionsPanel.py
#
# Copyright David Baddeley, 2010
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
# import pylab
import weakref

class OverlayPanel(wx.Panel):
    def __init__(self, parent, view, mdh, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent, **kwargs)

        #self.parent = parent
        self.view = weakref.proxy(view)
        self.mdh = mdh

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.cbShowPoints = wx.CheckBox(self, -1, 'Localisations')
        self.cbShowPoints.SetValue(self.view.showPoints)
        vsizer.Add(self.cbShowPoints, 0, wx.ALL|wx.ALIGN_LEFT, 5)
        self.cbShowPoints.Bind(wx.EVT_CHECKBOX, self.OnShowPoints)

        self.cbShowAdjPoints = wx.CheckBox(self, -1, 'Adjacent Locs')
        self.cbShowAdjPoints.SetValue(self.view.showAdjacentPoints)
        vsizer.Add(self.cbShowAdjPoints, 0, wx.ALL|wx.ALIGN_LEFT, 5)
        self.cbShowAdjPoints.Bind(wx.EVT_CHECKBOX, self.OnShowAdjPoints)

        self.cbShiftCorr = wx.CheckBox(self, -1, 'Splitter Shift Corr')
        self.cbShiftCorr.SetValue('chroma' in dir(self.view))
        vsizer.Add(self.cbShiftCorr, 0, wx.ALL|wx.ALIGN_LEFT, 5)
        self.cbShiftCorr.Bind(wx.EVT_CHECKBOX, self.OnShiftCorr)

        self.SetSizer(vsizer)


    def OnShowPoints(self, event):
        self.view.showPoints = self.cbShowPoints.GetValue()
        self.view.Refresh()

    def OnShowAdjPoints(self, event):
        self.view.showAdjacentPoints = self.cbShowAdjPoints.GetValue()
        self.view.Refresh()

    def OnShiftCorr(self, event):
        if self.cbShiftCorr.GetValue():
            self.view.chroma = self.mdh.chroma
            self.view.vox_x = self.mdh.voxelsize.x
            self.view.vox_y = self.mdh.voxelsize.y
        else:
            if 'chroma' in dir(self.view):
                del self.view.chroma
                
        self.view.Refresh()



   


