#!/usr/bin/python

##################
# ArrayFrame.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Frame:ArrayVFrame

import wx
import ArrayViewPanel

def create(parent):
    return ArrayVFrame(parent)

[wxID_ARRAYVFRAME, wxID_ARRAYVFRAMEPANEL1, 
] = [wx.NewId() for _init_ctrls in range(2)]

class ArrayVFrame(wx.Frame):
    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizer1 = wx.BoxSizer(orient=wx.VERTICAL)

        self._init_coll_boxSizer1_Items(self.boxSizer1)

        self.SetSizer(self.boxSizer1)


    def _init_coll_boxSizer1_Items(self, parent):
        # generated method, don't edit

        parent.AddWindow(self.viewPanel, 0, border=0, flag=0)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_ARRAYVFRAME, name='ArrayVFrame',
              parent=prnt, pos=wx.Point(242, 262), size=wx.Size(504, 459),
              style=wx.DEFAULT_FRAME_STYLE, title='numpy View3D')
        self.SetClientSize(wx.Size(496, 432))

        self.viewPanel = ArrayViewPanel.ArrayViewPanel(id=-1,
              name='viewPanel', parent=self, pos=wx.Point(0, 0),
              size=wx.Size(496, 432), style=wx.TAB_TRAVERSAL)

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)
