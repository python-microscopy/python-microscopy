#!/usr/bin/python

##################
# editFilterDialog.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx


class FilterEditDialog(wx.Dialog):
    def __init__(self, parent, mode='new', possibleKeys=[], key="", minVal=0.0, maxVal=1e5):
        wx.Dialog.__init__(self, parent, title='Edit Filter ...')

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        
        sizer2.Add(wx.StaticText(self, -1, 'Key:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        self.cbKey = wx.ComboBox(self, -1, value=key, choices=possibleKeys, style=wx.CB_DROPDOWN, size=(150, -1))

        if not mode == 'new':
            self.cbKey.Enable(False)

        sizer2.Add(self.cbKey, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(wx.StaticText(self, -1, 'Min:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.tMin = wx.TextCtrl(self, -1, '%3.2f' % minVal, size=(60, -1))
        

        sizer2.Add(self.tMin, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(wx.StaticText(self, -1, 'Max:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.tMax = wx.TextCtrl(self, -1, '%3.2f' % maxVal, size=(60, -1))

        sizer2.Add(self.tMax, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        
        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)
        
        

        
        
