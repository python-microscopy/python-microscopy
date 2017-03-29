#!/usr/bin/python

# VideoPanel.py
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

#  template: DSView/OptionsPanel(wx.Panel):
import json

import cv2
import numpy
from wx import wx

import PYME.ui.autoFoldPanel as afp
from PYME.LMVis.View import View


class VideoPanel(wx.Panel):

    JSON_LIST_NAME = 'views'

    def __init__(self, parent, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent, **kwargs)
        self.views = []
        self.parent = parent
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)
        self.view_table = wx.ListCtrl(self, -1, style=wx.BU_EXACTFIT | wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.SUNKEN_BORDER)

        self.view_table.InsertColumn(0, 'id')

        self.view_table.SetColumnWidth(0, wx.LIST_AUTOSIZE_USEHEADER)
        vertical_sizer.Add(self.view_table)

        self.create_buttons(vertical_sizer)

        self.SetSizerAndFit(vertical_sizer)

    def get_canvas(self):
        return self.parent.glCanvas

    def create_buttons(self, vertical_sizer):
        grid_sizer = wx.GridSizer(3, 3)
        # generate the buttons
        add_button = wx.Button(self, -1, label='Add', style=wx.BU_EXACTFIT)
        delete_button = wx.Button(self, -1, label='Delete', style=wx.BU_EXACTFIT)
        skip = wx.StaticText(self, -1, '')
        load_button = wx.Button(self, -1, label='Load',style=wx.BU_EXACTFIT)
        save_button = wx.Button(self, -1, label='Save', style=wx.BU_EXACTFIT)
        clear_button = wx.Button(self, -1, label='Clear', style=wx.BU_EXACTFIT)
        run_button = wx.Button(self, -1, label='Run', style=wx.BU_EXACTFIT)


        # bind the buttons and its handlers
        self.Bind(wx.EVT_BUTTON, self.add_snapshot, add_button)
        self.Bind(wx.EVT_BUTTON, self.delete_snapshot, delete_button)
        self.Bind(wx.EVT_BUTTON, self.clear, clear_button)
        self.Bind(wx.EVT_BUTTON, self.load, load_button)
        self.Bind(wx.EVT_BUTTON, self.save, save_button)
        self.Bind(wx.EVT_BUTTON, self.run, run_button)

        # add_snapshot the buttons to the view
        grid_sizer.Add(add_button, flag=wx.EXPAND)
        grid_sizer.Add(delete_button, flag=wx.EXPAND)
        grid_sizer.Add(skip)
        grid_sizer.Add(load_button, flag=wx.EXPAND)
        grid_sizer.Add(save_button, flag=wx.EXPAND)
        grid_sizer.Add(clear_button, flag=wx.EXPAND)
        grid_sizer.Add(run_button, flag=wx.EXPAND)
        vertical_sizer.Add(grid_sizer)

    def add_snapshot_to_view(self, view):
        index = len(self.views)
        self.view_table.InsertStringItem(index, "test{}".format(index))
        self.views.append(index)

    def add_snapshot(self, event):
        self.add_snapshot_to_view(self.get_canvas().get_view())

    def delete_snapshot(self, event):
        index = self.view_table.GetFirstSelected()
        self.view_table.DeleteItem(index)
        self.views.remove(index)

    def clear(self, event):
        self.views = []
        self.view_table.DeleteAllItems()

    def save(self, event):
        file_name = wx.FileSelector('Save view as json named... ')
        if file_name:
            if not file_name.endswith('.json'):
                file_name = '{}.json'.format(file_name)
            with open(file_name, 'w') as f:
                f.write('{')
                f.write('\"{}\":['.format(self.JSON_LIST_NAME))
                is_first = True
                for view in self.views:
                    if not is_first:
                        f.write(',')
                    f.writelines(view.to_json())
                    is_first = False
                f.write(']}')

    def load(self, event):
        file_name = wx.FileSelector('Open View-JSON file')
        if file_name:
            with open(file_name, 'r') as f:
                data = json.load(f)
                for view in data[self.JSON_LIST_NAME]:
                    self.views.append(View.decode_json(view))

    def run(self, event):
        width = self.get_canvas().Size[0]
        height = self.get_canvas().Size[1]
        self.get_canvas().displayMode = '3D'
        file_name = wx.FileSelector('Save video as avi named... ')

        if file_name:
            if not file_name.endswith('.avi'):
                file_name = '{}.avi'.format(file_name)
            video = cv2.VideoWriter(file_name, -1, 30, (width, height))
            if not self.views:
                self.add_view(self.get_canvas())
            current_view = None
            for view in self.views:
                if not current_view:
                    current_view = view
                else:
                    steps = 40
                    difference_view = (view - current_view) / steps
                    for step in range(0, steps):
                        new_view = current_view + difference_view * step
                        self.get_canvas().set_view(new_view)
                        img = numpy.fromstring(self.get_canvas().getIm().tostring(), numpy.ubyte).reshape(height, width, 3)
                        video.write(cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR))
                    current_view = view
            video.release()


class VideoFrame(wx.Frame):

    def __init__(self, parent_frame):
        wx.Frame.__init__(self, parent_frame, title='Display')

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.dispPanel = VideoPanel(self)

        hsizer.Add(self.dispPanel, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)

def Plug(visFr):
    pass

if __name__ == '__main__':
    app = wx.PySimpleApp()
    parent = VideoFrame(None, -1)
    parent.Show()
    app.MainLoop()
