#!/usr/bin/python

##################
# arclampshutterpanel.py
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

class Arclampshutterpanel(wx.Panel):
    def __init__(self, parent, lampshutter = None, winid=-1):
        # begin wxGlade: MyFrame1.__init__
        #kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Panel.__init__(self, parent, winid)
        self.lampshutter = lampshutter

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        n = 0

        self.cBoxes = []

        for l in self.lampshutter:
            cb = wx.CheckBox(self, -1, l.GetName())
            cb.SetValue(l.IsOn())
            cb.Bind(wx.EVT_CHECKBOX, self.OnCbOn)
            self.cBoxes.append(cb)
            hsizer.Add(cb,1, wx.EXPAND, 0)
            n += 1
            if (n % 3) == 0:
                sizer_1.Add(hsizer,0, wx.EXPAND, 0)
                hsizer = wx.BoxSizer(wx.HORIZONTAL)

        sizer_1.Add(hsizer,0, wx.EXPAND, 0)


        self.SetAutoLayout(1)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        sizer_1.SetSizeHints(self)
        self.Layout()
        # end wxGlade

    def OnCbOn(self, event):
        cb = event.GetEventObject()
        ind = self.cBoxes.index(cb)

        if cb.GetValue():
            self.lampshutter[ind].TurnOn()
        else:
            self.lampshutter[ind].TurnOff()

    def refresh(self):
        for l, cb in zip(self.lampshutter, self.cBoxes):
            cb.SetValue(l.IsOn())
