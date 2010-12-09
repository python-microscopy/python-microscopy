#!/usr/bin/python

##################
# deClumpGUI.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
#import deClump
from pylab import *

class deClumpDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Clump Radius: '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tClumpRadMult = wx.TextCtrl(self, -1, '2.0', size=[30,-1])
        hsizer.Add(self.tClumpRadMult, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'X'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cClumpRadVar = wx.Choice(self, -1, choices=['1.0', 'error_x'])
        self.cClumpRadVar.SetSelection(1)
        hsizer.Add(self.cClumpRadVar,1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        vsizer.Add(hsizer, 0, wx.ALL, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Time Window: '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tClumpTime = wx.TextCtrl(self, -1, '3')
        hsizer.Add(self.tClumpTime,1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        vsizer.Add(btSizer, 0, wx.ALL, 5)

        self.SetSizerAndFit(vsizer)

    def GetClumpRadiusMultiplier(self):
        return float(self.tClumpRadMult.GetValue())

    def GetClumpRadiusVariable(self):
        return self.cClumpRadVar.GetStringSelection()

    def GetClumpTimeWindow(self):
        return int(self.tClumpTime.GetValue())





