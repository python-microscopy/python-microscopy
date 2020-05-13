#!/usr/bin/python

# showShortcuts.py.py
#
# Copyright Michael Graff
#   graff@hm.edu
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
import wx

shortcuts = [('Focus on the histogramm + P', 'Set 10% of the values below/above limits'),
             ('Focus on the canvas + C', 'Center the model in the canvas'),
             ('Focus on the canvas + 3D + S', 'Enable stereo mode'),
             ('Focus on the canvas + stereo + [', 'Reduce the stereo distance'),
             ('Focus on the canvas + stereo + ]', 'Increase the stereo distance'),
             ('Focus on the canvas + R', 'Reset the rotation (translation with C)')]

class ShowShortcuts(wx.Dialog):
    def __init__(self, *args, **kwargs):
        super(ShowShortcuts, self).__init__(*args, **kwargs)

        sizer = self.init_pane()
        self.SetSizerAndFit(sizer)
        self.SetTitle('Known Shortcuts')

    def init_pane(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        table = wx.FlexGridSizer(0, 2, 4, 20)
        for pair in shortcuts:
            self.add_shortcut(table, pair[0], pair[1])

        ok_button = wx.Button(self, label='Ok')
        ok_button.Bind(wx.EVT_BUTTON, self.OnClose)
        sizer.Add(table, border=5, flag= wx.ALL)
        sizer.Add(ok_button, flag=wx.ALIGN_CENTER)
        return sizer

    def add_shortcut(self, table, short_cut, action):
        table.Add(wx.StaticText(self, label=short_cut))
        table.Add(wx.StaticText(self, label=action))

    def OnClose(self, e):
        if self.IsModal():
            self.EndModal(e.EventObject.Id)
        else:
            self.Destroy()

def Plug(vis_fr):

    vis_fr.AddMenuItem('Help', 'Show Shortcuts', lambda e: show_shortcuts(vis_fr))


def show_shortcuts(vis_fr):
    shortcuts = ShowShortcuts(vis_fr)
    shortcuts.ShowModal()
    shortcuts.Destroy()
