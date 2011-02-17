#!/usr/bin/python
##################
# selectCameraPanel.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
# GUI controls for changing the camera
#
##################

import wx


class CameraChooserPanel(wx.Panel):
    def __init__(self, parent, scope, **kwargs):
        wx.Panel.__init__(self, parent, **kwargs)

        self.scope = scope

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cCamera = wx.Choice(self, -1, choices = scope.cameras.keys())
        self.cCamera.SetSelection(scope.cameras.values().index(scope.cam))
        self.cCamera.Bind(wx.EVT_CHOICE, self.OnCCamera)

        hsizer.Add(self.cCamera, 1, wx.ALL, 2)

        self.SetSizerAndFit(hsizer)

    def OnCCamera(self, event):
        self.scope.SetCamera(self.cCamera.GetStringSelection())
