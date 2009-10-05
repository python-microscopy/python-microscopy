#!/usr/bin/python

##################
# FrFilter.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Frame:FiltFrame

import wx

def create(parent):
    return FiltFrame(parent)

[wxID_FILTFRAME, wxID_FILTFRAMECHFILTWHEEL, wxID_FILTFRAMEPANEL1, 
] = [wx.NewId() for _init_ctrls in range(3)]

class FiltPanel(wx.Frame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, prnt, -1, size=wx.Size(156, 65))
        self.SetClientSize(wx.Size(148, 38))


        self.chFiltWheel = wx.Choice(choices=[], id=wxID_FILTFRAMECHFILTWHEEL,
              name=u'chFiltWheel', parent=self, pos=wx.Point(8, 8),
              size=wx.Size(128, 21), style=0)
        self.chFiltWheel.Bind(wx.EVT_CHOICE, self.OnChFiltWheelChoice,
              id=wxID_FILTFRAMECHFILTWHEEL)

    def __init__(self, parent):
        self._init_ctrls(parent)

    def OnChFiltWheelChoice(self, event):
        event.Skip()
