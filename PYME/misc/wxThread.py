#!/usr/bin/python

##################
# wxThread.py
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

import wx
import threading

class wxThread(threading.Thread):
    def run(self):
        self.app = wx.PySimpleApp()
        self.f1 = wx.Frame(None,-1,'ball_wx',wx.DefaultPosition,wx.Size(400,400))
        self.f1.Show()
        #self.f1.Iconize()
        self.app.MainLoop()