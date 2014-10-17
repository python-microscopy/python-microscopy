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
        
        self.cbTrack = wx.CheckBox(self, -1, 'Track')
        hsizer.Add(self.cbTrack, 0, wx.ALL, 2) 
        self.cbTrack.Bind(wx.EVT_CHECKBOX, self.OnCBTrack)
        self.cbLock = wx.CheckBox(self, -1, 'Lock')
        self.cbLock.Bind(wx.EVT_CHECKBOX, self.OnCBLock)
        hsizer.Add(self.cbLock, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bSetPostion = wx.Button(self, -1, 'Set focus to current')
        hsizer.Add(self.bSetPostion, 0, wx.ALL, 2) 
        self.bSetPostion.Bind(wx.EVT_BUTTON, self.OnBSetPostion)
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stError = wx.StaticText(self, -1, 'Error:\n\n')
        hsizer.Add(self.stError, 0, wx.ALL, 2)        
        sizer_1.Add(hsizer,0, wx.EXPAND, 0)
        


        self.SetAutoLayout(1)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        sizer_1.SetSizeHints(self)
        self.Layout()
        # end wxGlade

    def OnCBTrack(self, event):
        #print self.cbTrack.GetValue()
        if self.cbTrack.GetValue():
            self.dt.register()
        else:
            self.dt.deregister()
            
    def OnBSetPostion(self, event):
        self.dt.reCalibrate()
        
    def OnCBLock(self, event):
        self.dt.lockFocus = self.cbLock.GetValue()

    def refresh(self):
        try:
            t, dx, dy, dz = self.dt.history[-1]
            self.stError.SetLabel('Error: x = %3.2f px\ny = %3.2f px\nz = %3.2f nm' % (dx, dy, dz*1000))
        except AttributeError:
            pass
        except IndexError:
            pass