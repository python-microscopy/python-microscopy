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

import os
import json
from time import sleep

import wx
import wx.lib.agw.aui as aui

from PIL import Image

from PYME.LMVis.Extras.dockedPanel import DockedPanel
from PYME.LMVis.views import VideoView

# See PYME.LMVis.Extras.dockedPanel for a history of afp import name
import PYME.ui.manualFoldPanel as afp

# Export file types
EXPORT_FILE_TYPES = ['JPG', 'PNG', 'TIFF']

# noinspection PyUnusedLocal
class VideoPanel(DockedPanel):
    JSON_LIST_NAME = 'views'

    def __init__(self, parent_panel, **kwargs):
        DockedPanel.__init__(self, parent_panel, **kwargs)

        self.snapshots = list()
        self.next_view_id = 0

        self.AddNewElement(self._anim_pan())

        clp = afp.collapsingPane(self, caption='Settings ...')
        clp.AddNewElement(self._settings_pan(clp))
        self.AddNewElement(clp)

    def _anim_pan(self):
        pan = wx.Panel(parent=self, style=wx.TAB_TRAVERSAL)
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)
        self.view_table = wx.ListCtrl(pan, -1,
                                      style=wx.BU_EXACTFIT | wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.SUNKEN_BORDER)

        self.view_table.InsertColumn(0, '#')
        self.view_table.InsertColumn(1, 'Name')
        self.view_table.InsertColumn(2, 'Duration')

        self.view_table.SetColumnWidth(0, 30)
        self.view_table.SetColumnWidth(1, 60)
        self.view_table.SetColumnWidth(2, 60)
        vertical_sizer.Add(self.view_table, 0, wx.EXPAND, 0)

        vertical_sizer.AddSpacer(10)

        self._create_buttons(pan, vertical_sizer)

        pan.SetSizerAndFit(vertical_sizer)

        return pan
    
    def _settings_pan(self, clp):
        pan = wx.Panel(clp, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.file_type = wx.ComboBox(pan, -1, choices=EXPORT_FILE_TYPES, style=wx.CB_DROPDOWN)
        hsizer.Add(wx.StaticText(pan, -1, 'Export file type:'), 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        hsizer.Add(self.file_type, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # self.pixel_size = wx.TextCtrl(pan, -1, size=(60, -1), value='5.0')
        # hsizer.Add(wx.StaticText(pan, -1, 'Pixel size:'), 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        # hsizer.Add(self.pixel_size, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        vsizer.Add(hsizer, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 0)
        pan.SetSizerAndFit(vsizer)
        return pan

    def _create_buttons(self, pan, vertical_sizer):
        grid_sizer = wx.GridSizer(rows=3, cols=3, vgap=2, hgap=2)
        # generate the buttons
        add_button = wx.Button(pan, -1, label='Add', style=wx.BU_EXACTFIT)
        delete_button = wx.Button(pan, -1, label='Delete', style=wx.BU_EXACTFIT)
        load_button = wx.Button(pan, -1, label='Load', style=wx.BU_EXACTFIT)
        save_button = wx.Button(pan, -1, label='Save', style=wx.BU_EXACTFIT)
        clear_button = wx.Button(pan, -1, label='Clear', style=wx.BU_EXACTFIT)
        run_button = wx.Button(pan, -1, label=u'\u25B7', style=wx.BU_EXACTFIT)
        make_button = wx.Button(pan, -1, label='Capture', style=wx.BU_EXACTFIT)

        # bind the buttons and its handlers
        self.Bind(wx.EVT_BUTTON, self.add_snapshot, add_button)
        self.Bind(wx.EVT_BUTTON, self.delete_snapshot, delete_button)
        self.Bind(wx.EVT_BUTTON, self.clear, clear_button)
        self.Bind(wx.EVT_BUTTON, self.load, load_button)
        self.Bind(wx.EVT_BUTTON, self.save, save_button) 
        self.Bind(wx.EVT_BUTTON, self.run, run_button)
        self.Bind(wx.EVT_BUTTON, self.make, make_button)

        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_edit)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_select_view)

        # add_snapshot the buttons to the view
        grid_sizer.Add(add_button, flag=wx.EXPAND)
        grid_sizer.Add(delete_button, flag=wx.EXPAND)
        grid_sizer.AddSpacer(0)
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
        #vec_id = self.ask(self, message='Please enter view id')
        vec_id = 'view_%d' % self.next_view_id
        self.next_view_id += 1
        if vec_id:
            duration = 3.0
            view = self.get_canvas().get_view(vec_id)
            video_view = VideoView(view.view_id,
                                   view.vec_up,
                                   view.vec_back,
                                   view.vec_right,
                                   view.translation,
                                   view.scale,
                                   view.clipping,
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
        file_name = wx.FileSelector('Save view as json named... ', flags=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
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
        # Fail gracefully if there are no snapshots
        if len(self.snapshots) == 0:
            wx.MessageBox('Need views to generate animation! Please create some by pressing `Add` in the animation pane.', 'No animation views', wx.OK | wx.ICON_WARNING)
            return
        # width = self.get_canvas().Size[0]
        # height = self.get_canvas().Size[1]
        self.get_canvas().displayMode = '3D'
        fps = 30.0
        # file_name = None
        
        # if save:
        #     try:
        #         import cv2
        #     except ImportError:
        #         msg_text = 'OpenCV 2 is needed to save videos. Please install: \"conda install -c menpo opencv\"'
        #         msg = wx.MessageDialog(self, msg_text, 'Missing package', wx.OK | wx.ICON_WARNING)
        #         msg.ShowModal()
        #         msg.Destroy()
        #         #return
        #         save=False

        #dir_name = None
        if save:
            #file_name = wx.FileSelector('Save video as avi named... ', flags=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
            #if file_name and not file_name.endswith('.avi'):
            #    file_name = '{}.avi'.format(file_name)
            dir_name = wx.DirSelector()
            if dir_name == '':
                return
            
            
        # video = None
        if True:
            # if save:
            #     pass
            #     #video = cv2.VideoWriter(file_name, -1, fps, (width, height))
            
            f_no = 0
                
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
                        self.get_canvas().set_view(new_view.normalize_view())
                        if save:
                            snap = self.get_canvas().getIm() #pixel_size=float(self.pixel_size.GetValue()))#.astype('uint8')
                            #print snap.shape, snap.dtype, snap.min(), snap.max()
                            if snap.dtype == 'uint8':
                                # PYME.LMVis.gl_render3D_shaders already returns 
                                # a snapshot as an 8-bit array
                                snap = snap.transpose(1,0,2)
                            else:
                                snap = (255*snap).astype('uint8').transpose(1, 0, 2)
                            
                            #if snap.ndim == 3:
                            #    video.write(cv2.cvtColor(cv2.flip(snap.transpose(1, 0, 2), 0), cv2.COLOR_RGB2BGR))
                            #else:
                            #    video.write(cv2.flip(snap.transpose(1, 0, 2), 0))
                            im = Image.fromarray(snap).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(dir_name, 
                                                            'frame{:04d}.{}'.format(f_no,self.file_type.GetValue().lower())))
                            f_no += 1
                        else:
                            # sleep(2.0/steps)
                            sleep(view.duration/steps)
                            self.get_canvas().OnDraw()
                    current_view = view
            if save:
                #video.release()
                msg_text = 'Video generation finished'
                msg = wx.MessageDialog(self, msg_text, 'Done', wx.OK | wx.ICON_INFORMATION)
                msg.ShowModal()
                msg.Destroy()

    # @staticmethod
    # def ask(parent=None, message='', default_value=''):
    #     dlg = wx.TextEntryDialog(parent, message, default_value)
    #     dlg.ShowModal()
    #     result = dlg.GetValue()
    #     dlg.Destroy()
    #     return result

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

    def on_select_view(self, event):
        snapshot = self.snapshots[self.view_table.GetFirstSelected()]

        self.get_canvas().set_view(snapshot)

    def refill(self):
        self.clear_view()
        for index, snapshot in enumerate(self.snapshots):
            # index = len(self.snapshots)
            # NOTE: InsertStringItem and SetStringItem are deprecated in wx > 4.0 favour of InsertItem and SetItem. Using old methods for wx=3.x compatibility.

            self.view_table.InsertStringItem(index, str(index))
            self.view_table.SetStringItem(index, 1, snapshot.view_id)
            self.view_table.SetStringItem(index, 2, "{:.9f}".format(snapshot.duration))


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
        grid_sizer = wx.GridSizer(rows=2, cols=2, vgap=1, hgap=1)

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
