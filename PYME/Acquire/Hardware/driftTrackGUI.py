#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import wx

class DriftTrackingControl(wx.Panel):
    def __init__(self, parent, driftTracker, winid=-1):
        # begin wxGlade: MyFrame1.__init__
        #kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Panel.__init__(self, parent, winid)
        self.dt = driftTracker

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        bStart =

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
            self.lasers[ind].TurnOn()
        else:
            self.lasers[ind].TurnOff()

    def refresh(self):
        for l, cb in zip(self.lasers, self.cBoxes):
            cb.SetValue(l.IsOn())