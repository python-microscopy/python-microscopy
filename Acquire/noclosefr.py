#!/usr/bin/python

##################
# noclosefr.py
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

class noCloseFrame(wx.Frame):
    def __init__(self,*args, **kwds):
        wx.Frame.__init__(self,*args, **kwds)
        wx.EVT_CLOSE(self, self.OnCloseWindow)

    def OnCloseWindow(self, event):   
        if (not event.CanVeto()): 
            self.Destroy()
        else:
            event.Veto()
            self.Hide()
            
class wxFrame(wx.Frame):
    def __init__(self,*args, **kwds):
        wx.Frame.__init__(self,*args, **kwds)
        wx.EVT_CLOSE(self, self.OnCloseWindow)

    def OnCloseWindow(self, event):   
        if (not event.CanVeto()): 
            self.Destroy()
        else:
            event.Veto()
            self.Hide()