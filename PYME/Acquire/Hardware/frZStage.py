#!/usr/bin/python

##################
# frZStage.py
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

#Boa:Frame:frZStepper

import wx

def create(parent):
    return frZStepper(parent)

[wxID_FRZSTEPPER, wxID_FRZSTEPPERBMINUS1U, wxID_FRZSTEPPERBMINUS200N, 
 wxID_FRZSTEPPERBMINUS50N, wxID_FRZSTEPPERBPLUS1U, wxID_FRZSTEPPERBPLUS200N, 
 wxID_FRZSTEPPERBPLUS50N, wxID_FRZSTEPPERPANEL1, wxID_FRZSTEPPERTCCURPOS, 
] = [wx.NewIdRef() for _init_ctrls in range(9)]

class frZStepper(wx.Frame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRZSTEPPER, name='frZStepper',
              parent=prnt, pos=wx.Point(426, 453), size=wx.Size(126, 161),
              style=wx.DEFAULT_FRAME_STYLE, title='ZStage')
        self.SetClientSize(wx.Size(118, 134))

        self.panel1 = wx.Panel(id=wxID_FRZSTEPPERPANEL1, name='panel1',
              parent=self, pos=wx.Point(0, 0), size=wx.Size(118, 134),
              style=wx.TAB_TRAVERSAL)

        self.bPlus50n = wx.Button(id=wxID_FRZSTEPPERBPLUS50N, label='+50nm',
              name='bPlus50n', parent=self.panel1, pos=wx.Point(8, 72),
              size=wx.Size(48, 23), style=0)
        self.bPlus50n.Bind(wx.EVT_BUTTON, self.OnBPlus50nButton,
              id=wxID_FRZSTEPPERBPLUS50N)

        self.bPlus200n = wx.Button(id=wxID_FRZSTEPPERBPLUS200N, label='+200nm',
              name='bPlus200n', parent=self.panel1, pos=wx.Point(8, 41),
              size=wx.Size(48, 23), style=0)
        self.bPlus200n.Bind(wx.EVT_BUTTON, self.OnBPlus200nButton,
              id=wxID_FRZSTEPPERBPLUS200N)

        self.bMinus1u = wx.Button(id=wxID_FRZSTEPPERBMINUS1U, label='-1um',
              name='bMinus1u', parent=self.panel1, pos=wx.Point(64, 8),
              size=wx.Size(48, 23), style=0)
        self.bMinus1u.Bind(wx.EVT_BUTTON, self.OnBMinus1uButton,
              id=wxID_FRZSTEPPERBMINUS1U)

        self.tcCurPos = wx.TextCtrl(id=wxID_FRZSTEPPERTCCURPOS, name='tcCurPos',
              parent=self.panel1, pos=wx.Point(8, 104), size=wx.Size(88, 21),
              style=0, value='textCtrl1')
        self.tcCurPos.SetEditable(False)

        self.bMinus200n = wx.Button(id=wxID_FRZSTEPPERBMINUS200N,
              label='-200nm', name='bMinus200n', parent=self.panel1,
              pos=wx.Point(64, 40), size=wx.Size(48, 23), style=0)
        self.bMinus200n.Bind(wx.EVT_BUTTON, self.OnBMinus200nButton,
              id=wxID_FRZSTEPPERBMINUS200N)

        self.bMinus50n = wx.Button(id=wxID_FRZSTEPPERBMINUS50N, label='-50nm',
              name='bMinus50n', parent=self.panel1, pos=wx.Point(64, 72),
              size=wx.Size(48, 23), style=0)
        self.bMinus50n.Bind(wx.EVT_BUTTON, self.OnBMinus50nButton,
              id=wxID_FRZSTEPPERBMINUS50N)

        self.bPlus1u = wx.Button(id=wxID_FRZSTEPPERBPLUS1U, label='+1um',
              name='bPlus1u', parent=self.panel1, pos=wx.Point(8, 8),
              size=wx.Size(48, 23), style=0)
        self.bPlus1u.Bind(wx.EVT_BUTTON, self.OnBPlus1uButton,
              id=wxID_FRZSTEPPERBPLUS1U)

    def __init__(self, parent, piezo):
        self._init_ctrls(parent)
        self.piezo = piezo
        self.update()
        
    def update(self):
        self.tcCurPos.SetValue('%f' %self.piezo.GetPos())
    
    def moveRel(self, dist):
        self.piezo.MoveTo(0, self.piezo.GetPos() + dist)
        self.update()

    def OnBPlus50nButton(self, event):
        self.moveRel(0.05)

    def OnBPlus200nButton(self, event):
        self.moveRel(0.2)

    def OnBMinus1uButton(self, event):
        self.moveRel(-1)

    def OnBMinus200nButton(self, event):
        self.moveRel(-0.2)

    def OnBMinus50nButton(self, event):
        self.moveRel(-0.05)

    def OnBPlus1uButton(self, event):
        self.moveRel(1)
