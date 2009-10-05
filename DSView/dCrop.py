#!/usr/bin/python

##################
# dCrop.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#Boa:Dialog:dCrop

#from wxPython.wx. import *
import wx
from wx.lib.anchors import LayoutAnchors

def create(parent):
    return dCrop(parent)

[wxID_DCROP, wxID_DCROPBCANCEL, wxID_DCROPBCROP, wxID_DCROPCLBCHANS, 
 wxID_DCROPSTATICTEXT1, wxID_DCROPSTATICTEXT2, wxID_DCROPSTATICTEXT3, 
 wxID_DCROPSTATICTEXT4, wxID_DCROPSTATICTEXT5, wxID_DCROPSTATICTEXT6, 
 wxID_DCROPSTSIZE, wxID_DCROPTXEND, wxID_DCROPTXSTART, wxID_DCROPTYEND, 
 wxID_DCROPTYSTART, wxID_DCROPTZEND, wxID_DCROPTZSTART, 
] = map(lambda _init_ctrls: wx.NewId(), range(17))

class dCrop(wx.Dialog):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Dialog.__init__(self, id=wxID_DCROP, name='dCrop', parent=prnt,
              pos=wx.Point(460, 318), size=wx.Size(358, 206),
              style=wx.DEFAULT_DIALOG_STYLE, title='Crop')
        self.SetClientSize(wx.Size(350, 179))

        self.clbChans = wx.CheckListBox(choices=[], id=wxID_DCROPCLBCHANS,
              name='clbChans', parent=self, pos=wx.Point(192, 40),
              size=wx.Size(136, 88), style=0)
        EVT_KILL_FOCUS(self.clbChans, self.OnFocusChange)
        EVT_CHECKLISTBOX(self.clbChans, wxID_DCROPCLBCHANS, self.OnFocusChange)

        self.tYStart = wx.TextCtrl(id=wxID_DCROPTYSTART, name='tYStart',
              parent=self, pos=wx.Point(40, 72), size=wx.Size(56, 21), style=0,
              value='0')
        EVT_KILL_FOCUS(self.tYStart, self.OnFocusChange)

        self.tXStart = wx.TextCtrl(id=wxID_DCROPTXSTART, name='tXStart',
              parent=self, pos=wx.Point(40, 40), size=wx.Size(56, 21), style=0,
              value='0')
        EVT_KILL_FOCUS(self.tXStart, self.OnFocusChange)

        self.tXEnd = wx.TextCtrl(id=wxID_DCROPTXEND, name='tXEnd', parent=self,
              pos=wx.Point(112, 40), size=wx.Size(56, 21), style=0, value='0')
        EVT_KILL_FOCUS(self.tXEnd, self.OnFocusChange)

        self.tYEnd = wx.TextCtrl(id=wxID_DCROPTYEND, name='tYEnd', parent=self,
              pos=wx.Point(112, 72), size=wx.Size(56, 21), style=0, value='0')
        EVT_KILL_FOCUS(self.tYEnd, self.OnFocusChange)

        self.tZStart = wx.TextCtrl(id=wxID_DCROPTZSTART, name='tZStart',
              parent=self, pos=wx.Point(40, 104), size=wx.Size(56, 21), style=0,
              value='0')
        EVT_KILL_FOCUS(self.tZStart, self.OnFocusChange)

        self.tZEnd = wx.TextCtrl(id=wxID_DCROPTZEND, name='tZEnd', parent=self,
              pos=wx.Point(112, 104), size=wx.Size(56, 21), style=0, value='0')
        EVT_KILL_FOCUS(self.tZEnd, self.OnFocusChange)

        self.staticText1 = wx.StaticText(id=wxID_DCROPSTATICTEXT1, label='Y',
              name='staticText1', parent=self, pos=wx.Point(16, 80),
              size=wx.Size(7, 13), style=0)
        self.staticText1.SetConstraints(LayoutAnchors(self.staticText1, False,
              True, False, False))

        self.staticText2 = wx.StaticText(id=wxID_DCROPSTATICTEXT2, label='Z',
              name='staticText2', parent=self, pos=wx.Point(16, 112),
              size=wx.Size(7, 13), style=0)
        self.staticText2.SetConstraints(LayoutAnchors(self.staticText2, False,
              True, False, False))

        self.staticText3 = wx.StaticText(id=wxID_DCROPSTATICTEXT3, label='Start',
              name='staticText3', parent=self, pos=wx.Point(56, 16),
              size=wx.Size(22, 13), style=0)
        self.staticText3.SetConstraints(LayoutAnchors(self.staticText3, False,
              True, False, False))

        self.staticText4 = wx.StaticText(id=wxID_DCROPSTATICTEXT4, label='End',
              name='staticText4', parent=self, pos=wx.Point(128, 16),
              size=wx.Size(19, 13), style=0)
        self.staticText4.SetConstraints(LayoutAnchors(self.staticText4, False,
              True, False, False))

        self.staticText5 = wx.StaticText(id=wxID_DCROPSTATICTEXT5,
              label='Channels', name='staticText5', parent=self,
              pos=wx.Point(240, 16), size=wx.Size(44, 13), style=0)
        self.staticText5.SetConstraints(LayoutAnchors(self.staticText5, False,
              True, False, False))

        self.staticText6 = wx.StaticText(id=wxID_DCROPSTATICTEXT6, label='X    ',
              name='staticText6', parent=self, pos=wx.Point(16, 40),
              size=wx.Size(19, 13), style=0)
        self.staticText6.SetConstraints(LayoutAnchors(self.staticText6, False,
              True, False, False))

        self.stSize = wx.StaticText(id=wxID_DCROPSTSIZE, label='Size:',
              name='stSize', parent=self, pos=wx.Point(16, 152), size=wx.Size(23,
              13), style=0)

        self.bCrop = wx.Button(id=wxID_DCROPBCROP, label='Crop', name='bCrop',
              parent=self, pos=wx.Point(168, 144), size=wx.Size(75, 23), style=0)
        EVT_BUTTON(self.bCrop, wxID_DCROPBCROP, self.OnBCrop)

        self.bCancel = wx.Button(id=wxID_DCROPBCANCEL, label='Cancel',
              name='bCancel', parent=self, pos=wx.Point(256, 144),
              size=wx.Size(75, 23), style=0)
        EVT_BUTTON(self.bCancel, wxID_DCROPBCANCEL, self.OnBCancel)

    def __init__(self, parent, vp):
        self._init_ctrls(parent)
        
        self.tXStart.SetValue('%d' % vp.selection_begin_x)
        self.tYStart.SetValue('%d' % vp.selection_begin_y)
        self.tZStart.SetValue('%d' % vp.selection_begin_z)
        self.tXEnd.SetValue('%d' % vp.selection_end_x)
        self.tYEnd.SetValue('%d' % vp.selection_end_y)
        self.tZEnd.SetValue('%d' % vp.selection_end_z)
        
        for i in range(vp.ds.getNumChannels()):
            self.clbChans.Append(vp.ds.getChannelName(i))
            self.clbChans.Check(i)
            
        self.OnFocusChange(None)

    def OnFocusChange(self, event):
        self.x1 = int(self.tXStart.GetValue())
        self.y1 = int(self.tYStart.GetValue())
        self.z1 = int(self.tZStart.GetValue())
        self.x2 = int(self.tXEnd.GetValue())
        self.y2 = int(self.tYEnd.GetValue())
        self.z2 = int(self.tZEnd.GetValue())
        
        self.chs = []
        for i in range(self.clbChans.GetCount()):
            if self.clbChans.IsChecked(i):
                self.chs.append(i)
        
        self.chs = tuple(self.chs)
        
        self.stSize.SetLabel('Size: %d x %d x %d x %d' % (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1, len(self.chs)))
        
        #event.Skip()

    def OnBCancel(self, event):
        self.EndModal(False)
        #event.Skip()

    def OnBCrop(self, event):
        self.OnFocusChange(None)
        self.EndModal(True)
        #event.Skip()
