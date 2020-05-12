#!/usr/bin/python

# ViewPanel.py
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
import json

import wx

from PYME.LMVis.Extras.dockedPanel import DockedPanel
from PYME.LMVis.views import View


class ViewPanel(DockedPanel):
    def __init__(self, parent_panel, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent_panel, **kwargs)
        self.parent_panel = parent_panel
        self.alternate = True
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)

        self.create_buttons(vertical_sizer)

        self.SetSizerAndFit(vertical_sizer)

    def create_buttons(self, vertical_sizer):
        grid_sizer = wx.GridSizer(rows=6, cols=1, vgap=1, hgap=1)
        x_button = wx.Button(self, -1, label='View in x', style=wx.BU_EXACTFIT)
        y_button = wx.Button(self, -1, label='View in y', style=wx.BU_EXACTFIT)
        z_button = wx.Button(self, -1, label='View in z', style=wx.BU_EXACTFIT)

        skip = wx.StaticText(self, -1, label='', style=wx.BU_EXACTFIT)
        save_button = wx.Button(self, -1, label='Save View', style=wx.BU_EXACTFIT)
        load_button = wx.Button(self, -1, label='Load View', style=wx.BU_EXACTFIT)

        self.Bind(wx.EVT_BUTTON, lambda e: self.set_view_x(), x_button)
        self.Bind(wx.EVT_BUTTON, lambda e: self.set_view_y(), y_button)
        self.Bind(wx.EVT_BUTTON, lambda e: self.set_view_z(), z_button)
        self.Bind(wx.EVT_BUTTON, lambda e: self.save_view(), save_button)
        self.Bind(wx.EVT_BUTTON, lambda e: self.load_view(), load_button)

        grid_sizer.Add(x_button, flag=wx.EXPAND)
        grid_sizer.Add(y_button, flag=wx.EXPAND)
        grid_sizer.Add(z_button, flag=wx.EXPAND)
        grid_sizer.Add(skip, flag=wx.EXPAND)
        grid_sizer.Add(save_button, flag=wx.EXPAND)
        grid_sizer.Add(load_button, flag=wx.EXPAND)
        vertical_sizer.Add(grid_sizer, flag=wx.EXPAND)

    def set_view_x(self):
        vec_up = [0, 1, 0]
        if self.alternate:
            vec_back = [1, 0, 0]
        else:
            vec_back = [-1, 0, 0]
        vec_right = [0, 0, 1]

        self.set_view(vec_up, vec_back, vec_right)

    def set_view_y(self):
        vec_up = [0, 0, 1]
        if self.alternate:
            vec_back = [0, 1, 0]
        else:
            vec_back = [0, -1, 0]
        vec_right = [1, 0, 0]
        self.set_view(vec_up, vec_back, vec_right)

    def set_view_z(self):
        vec_up = [0, 1, 0]
        if self.alternate:
            vec_back = [0, 0, 1]
        else:
            vec_back = [0, 0, -1]
        vec_right = [1, 0, 0]
        self.set_view(vec_up, vec_back, vec_right)

    def set_view(self, vec_up, vec_back, vec_right):
        if self.get_canvas().displayMode == '2D':
            dlg = wx.MessageDialog(self,
                                   'Setting views in only possible in 3d mode', 'Info', wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            old_view = self.get_canvas().get_view()
            self.alternate = not self.alternate
            new_view = View(old_view.view_id, vec_up, vec_back, vec_right, old_view.translation, old_view.scale)
            self.get_canvas().set_view(new_view)

    def save_view(self):
        view = self.get_canvas().get_view()
        file_name = wx.FileSelector('Save view as json named... ')
        if file_name:
            if not file_name.endswith('.json'):
                file_name = '{}.json'.format(file_name)
            with open(file_name, 'w') as f:
                f.writelines(json.dumps(view.to_json(), indent=4))

    def load_view(self):
        file_name = wx.FileSelector('Open View-JSON file')
        if file_name:
            with open(file_name, 'r') as f:
                view = View.decode_json(json.load(f))
            self.get_canvas().set_view(view)


def Plug(vis_fr):
    print('Saved Views are deprecated -  use the ones with animation instead and remove me!')
    pass
    #DockedPanel.add_menu_item(vis_fr, 'Saved Views', ViewPanel, 'view_panel')
