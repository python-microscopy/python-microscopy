#!/usr/bin/python

##################
# voxSizeDialog.py
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

class VoxSizeDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, title='Voxel Size:')

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, u'x\u00A0[\u00B5m]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tVoxX = wx.TextCtrl(self, -1, '0.08')

        sizer2.Add(self.tVoxX, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALL, 0)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, u'y\u00A0[\u00B5m]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tVoxY = wx.TextCtrl(self, -1, '0.08')

        sizer2.Add(self.tVoxY, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALL, 0)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, u'z\u00A0[\u00B5m]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tVoxZ = wx.TextCtrl(self, -1, '0.2')

        sizer2.Add(self.tVoxZ, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALL, 0)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

#        btn = wx.Button(self, wx.ID_CANCEL)
#
#        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def GetVoxX(self):
        return float(self.tVoxX.GetValue())

    def GetVoxY(self):
        return float(self.tVoxY.GetValue())

    def GetVoxZ(self):
        return float(self.tVoxZ.GetValue())

   