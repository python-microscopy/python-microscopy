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

import json
from time import sleep

import wx
import wx.lib.agw.aui as aui

from PYME.LMVis.Extras.dockedPanel_tmp import DockedPanel
from PYME.LMVis.views import VideoView


# noinspection PyUnusedLocal
class VideoPanel(wx.Panel):
    JSON_LIST_NAME = 'views'

    def __init__(self, parent_panel, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent_panel, **kwargs)
        self.snapshots = list()
        self.parent_panel = parent_panel
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)
        self.view_table = wx.ListCtrl(self, -1,
                                      style=wx.BU_EXACTFIT | wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.SUNKEN_BORDER)

        self.view_table.InsertColumn(0, 'id')

        self.view_table.SetColumnWidth(0, wx.LIST_AUTOSIZE_USEHEADER)
        vertical_sizer.Add(self.view_table)

        self.create_buttons(vertical_sizer)

        self.SetSizerAndFit(vertical_sizer)

    def get_canvas(self):
        return self.parent_panel.glCanvas

    def create_buttons(self, vertical_sizer):
        grid_sizer = wx.GridSizer(3, 3)
        # generate the buttons
        add_button = wx.Button(self, -1, label='Add', style=wx.BU_EXACTFIT)
        delete_button = wx.Button(self, -1, label='Delete', style=wx.BU_EXACTFIT)
        skip = wx.StaticText(self, -1, '')
        load_button = wx.Button(self, -1, label='Load', style=wx.BU_EXACTFIT)
        save_button = wx.Button(self, -1, label='Save', style=wx.BU_EXACTFIT)
        clear_button = wx.Button(self, -1, label='Clear', style=wx.BU_EXACTFIT)
        run_button = wx.Button(self, -1, label='Run', style=wx.BU_EXACTFIT)
        make_button = wx.Button(self, -1, label='Make', style=wx.BU_EXACTFIT)

        # bind the buttons and its handlers
        self.Bind(wx.EVT_BUTTON, self.add_snapshot, add_button)
        self.Bind(wx.EVT_BUTTON, self.delete_snapshot, delete_button)
        self.Bind(wx.EVT_BUTTON, self.clear, clear_button)
        self.Bind(wx.EVT_BUTTON, self.load, load_button)
        self.Bind(wx.EVT_BUTTON, self.save, save_button)
        self.Bind(wx.EVT_BUTTON, self.run, run_button)
        self.Bind(wx.EVT_BUTTON, self.make, make_button)

        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_edit)

        # add_snapshot the buttons to the view
        grid_sizer.Add(add_button, flag=wx.EXPAND)
        grid_sizer.Add(delete_button, flag=wx.EXPAND)
        grid_sizer.Add(skip)
        grid_sizer.Add(load_button, flag=wx.EXPAND)
        grid_sizer.Add(save_button, flag=wx.EXPAND)
        grid_sizer.Add(clear_button, flag=wx.EXPAND)
        grid_sizer.Add(run_button, flag=wx.EXPAND)
        grid_sizer.Add(make_button, flag=wx.EXPAND)
        vertical_sizer.Add(grid_sizer)

    def add_snapshot_to_list(self, snapshot):
        self.snapshots.append(snapshot)
        self.refill()

    def add_snapshot(self, event):
        vec_id = self.ask(self, message='Please enter view id')
        if vec_id:
            duration = 3.0
            view = self.get_canvas().get_view(vec_id)
            video_view = VideoView(view.view_id,
                                   view.vec_up,
                                   view.vec_back,
                                   view.vec_right,
                                   view.translation,
                                   view.zoom,
                                   duration)
            self.add_snapshot_to_list(video_view)

    def delete_snapshot(self, event):
        index = self.view_table.GetFirstSelected()
        if index >= 0:
            del self.snapshots[index]
        self.refill()

    def clear(self, event):
        self.snapshots = []
        self.view_table.DeleteAllItems()

    def clear_view(self):
        self.view_table.DeleteAllItems()

    def save(self, event):
        file_name = wx.FileSelector('Save view as json named... ')
        if file_name:
            if not file_name.endswith('.json'):
                file_name = '{}.json'.format(file_name)
            with open(file_name, 'w') as f:
                snapshots = [snapshot.to_json() for snapshot in self.snapshots]
                f.writelines(json.dumps({self.JSON_LIST_NAME: snapshots}, indent=4))

    def load(self, event):
        file_name = wx.FileSelector('Open View-JSON file')
        if file_name:
            with open(file_name, 'r') as f:
                data = json.load(f)
                for view in data[self.JSON_LIST_NAME]:
                    self.add_snapshot_to_list(VideoView.decode_json(view))

    def make(self, event):
        self.play(True)

    def run(self, event):
        self.play(False)

    def play(self, save):
        width = self.get_canvas().Size[0]
        height = self.get_canvas().Size[1]
        self.get_canvas().displayMode = '3D'
        fps = 30.0
        file_name = None
        try:
            import cv2
        except ImportError:
            msg_text = 'OpenCV 2 is needed to create videos. Please install: \"conda install -c menpo opencv\"'
            msg = wx.MessageDialog(self, msg_text, 'Missing package', wx.OK | wx.ICON_WARNING)
            msg.ShowModal()
            msg.Destroy()
            return

        if save:
            file_name = wx.FileSelector('Save video as avi named... ')
            if file_name and not file_name.endswith('.avi'):
                file_name = '{}.avi'.format(file_name)
        video = None
        if not save or file_name:
            if save:
                video = cv2.VideoWriter(file_name, -1, fps, (width, height))
            if not self.snapshots:
                self.add_snapshot_to_list(self.get_canvas())
            current_view = None
            for view in self.snapshots:
                if not current_view:
                    current_view = view
                else:
                    steps = int(round(view.duration * fps))
                    difference_view = (view - current_view) / steps
                    for step in range(0, steps):
                        new_view = current_view + difference_view * step
                        self.get_canvas().set_view(new_view)
                        if save:
                            snap = self.get_canvas().getIm()
                            if snap.ndim == 3:
                                video.write(cv2.cvtColor(cv2.flip(snap.transpose(1, 0, 2), 0), cv2.COLOR_RGB2BGR))
                            else:
                                video.write(cv2.flip(snap.transpose(1, 0, 2), 0))
                        else:
                            sleep(2.0/steps)
                            self.get_canvas().OnDraw()
                    current_view = view
            if save:
                video.release()
            msg_text = 'Video generation finished'
            msg = wx.MessageDialog(self, msg_text, 'Done', wx.OK | wx.ICON_INFORMATION)
            msg.ShowModal()
            msg.Destroy()

    @staticmethod
    def ask(parent=None, message='', default_value=''):
        dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
        dlg.ShowModal()
        result = dlg.GetValue()
        dlg.Destroy()
        return result

    # noinspection PyTypeChecker
    def on_edit(self, event):
        snapshot = self.snapshots[self.view_table.GetFirstSelected()]

        dlg = EditDialog(self, snapshot, 'Edit VideoView')
        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            name = dlg.get_name()
            duration = dlg.get_duration()

            snapshot.view_id = name
            snapshot.duration = duration

        dlg.Destroy()
        self.refill()

    def refill(self):
        self.clear_view()
        for snapshot in self.snapshots:
            index = len(self.snapshots)
            self.view_table.InsertStringItem(index, snapshot.view_id)


class EditDialog(wx.Dialog):
    def __init__(self, parent, snapshot, title=''):
        """

        Returns
        -------
        EditDialog
        """
        wx.Dialog.__init__(self, parent, title=title)

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.edit_panel = EditPanel(self, -1, snapshot, pos=(0, 0), size=(200, 100))
        sizer1.Add(self.edit_panel, 0, wx.ALL | wx.EXPAND, 5)

        # create button
        bt_sizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        bt_sizer.AddButton(btn)

        bt_sizer.Realize()

        sizer1.Add(bt_sizer, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        # set total size
        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def get_name(self):
        return self.edit_panel.id

    def get_duration(self):
        return self.edit_panel.duration


class EditPanel(wx.Panel):
    def __init__(self, parent, id_number, snapshot, size, pos):
        wx.Panel.__init__(self, parent, id_number, size=size, pos=pos, style=wx.BORDER_SUNKEN)
        grid_sizer = wx.GridSizer(2, 2)

        # generate row for view_id
        grid_sizer.Add(wx.StaticText(self, label='View Id', style=wx.BU_EXACTFIT))
        self.name_text = wx.TextCtrl(self, size=(100, -1), style=wx.BU_EXACTFIT)
        self.name_text.SetValue(snapshot.view_id)
        grid_sizer.Add(self.name_text)

        # generate row for duration
        grid_sizer.Add(wx.StaticText(self, label='Duration', style=wx.BU_EXACTFIT))
        self.duration_text = wx.TextCtrl(self, size=(100, -1), style=wx.BU_EXACTFIT)
        grid_sizer.Add(self.duration_text)
        self.duration_text.SetValue("{:.9f}".format(snapshot.duration))

        self.SetSizerAndFit(grid_sizer)

    @property
    def id(self):
        return self.name_text.GetValue()

    @property
    def duration(self):
        try:
            return float(self.duration_text.GetValue())
        except ValueError:
            return None


class VideoFrame(wx.Frame):
    def __init__(self, parent_frame):
        wx.Frame.__init__(self, parent_frame, title='Display')

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.dispPanel = VideoPanel(self)

        hsizer.Add(self.dispPanel, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)

def Plug(vis_fr):
    DockedPanel.add_menu_item(vis_fr, 'Animation', VideoPanel, 'animation_panel')
