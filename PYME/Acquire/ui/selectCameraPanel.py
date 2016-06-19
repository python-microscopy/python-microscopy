#!/usr/bin/python
##################
# selectCameraPanel.py
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


class CameraChooserPanel(wx.Panel):
    def __init__(self, parent, scope, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.scope = scope

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        cameraNames = list(scope.cameras.keys())

        self.cCamera = wx.Choice(self, -1, choices = cameraNames)
        self.cCamera.SetSelection(cameraNames.index(self.scope.state['ActiveCamera']))
        self.cCamera.Bind(wx.EVT_CHOICE, self.OnCCamera)

        hsizer.Add(self.cCamera, 1, wx.ALL, 2)

        self.SetSizerAndFit(hsizer)

    def OnCCamera(self, event):
        self.scope.state['ActiveCamera'] = self.cCamera.GetStringSelection()
        
