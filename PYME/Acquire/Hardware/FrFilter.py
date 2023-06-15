#!/usr/bin/python

##################
# FrFilter.py
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

#Boa:Frame:FiltFrame

import wx

def create(parent):
    return FiltFrame(parent)

[wxID_FILTFRAME, wxID_FILTFRAMECHFILTWHEEL, wxID_FILTFRAMEPANEL1, 
] = [wx.NewIdRef() for _init_ctrls in range(3)]

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
