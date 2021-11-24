# -*- coding: utf-8 -*-

"""
@author: zacsimile
"""

from wx.core import Image
from PYME.recipes.recipeGui import FileListPanel
from PYME.contrib import dispatch
from PYME.IO.DataSources import ConcatenatedDataSource
from PYME.IO.image import ImageStack

import wx

class SequencePanel(FileListPanel):
    def __init__(self, *args, **kwargs):
        FileListPanel.__init__(self, *args, **kwargs)
        self.on_update_file_list = dispatch.Signal()

    def UpdateFileList(self, filenames):
        FileListPanel.UpdateFileList(self, filenames)
        self.on_update_file_list.send(self)

class OpenSequenceDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, title="Open file sequence")

        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.file_panel = SequencePanel(self,-1)
        self._datasources = []
        vsizer.Add(self.file_panel)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Z:'), 0, wx.ALIGN_LEFT | wx.ALL, 5)
        self.size_z = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.size_z)
        hsizer.Add(wx.StaticText(self, -1, 'C:'), 0, wx.ALIGN_LEFT | wx.ALL, 5)
        self.size_c = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.size_c)
        hsizer.Add(wx.StaticText(self, -1, 'T:'), 0, wx.ALIGN_LEFT | wx.ALL, 5)
        self.size_t = wx.TextCtrl(self, -1, '0')
        hsizer.Add(self.size_t)

        vsizer.Add(hsizer, 0, wx.ALL, 0)

        self.btn_open = wx.Button(self, wx.ID_OK)
        btn_cancel = wx.Button(self, wx.ID_CANCEL)
        self.btn_open.SetDefault()

        btn_sizer = wx.StdDialogButtonSizer()
        btn_sizer.AddButton(btn_cancel)
        btn_sizer.AddButton(self.btn_open)
        btn_sizer.Realize()

        vsizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizerAndFit(vsizer)

        self.file_panel.on_update_file_list.connect(self.on_update)

    def on_update(self, **kwargs):
        self._datasources = []
        self.btn_open.Enable()
        for fn in self.file_panel.filenames:
            image = ImageStack(filename=fn)
            self._datasources.append(image.data_xyztc)
        
        try:
            self._datasource = ConcatenatedDataSource.DataSource(self._datasources)
            size_z, size_t, size_c = self._datasource._sizes
            self.size_z.SetValue(str(size_z))
            self.size_t.SetValue(str(size_t))
            self.size_c.SetValue(str(size_c))
        except(AssertionError):
            self.btn_open.Disable()

    def get_datasource(self):
        return ConcatenatedDataSource.DataSource(self._datasources, 
                                                 size_z=int(self.size_z.GetValue()), 
                                                 size_t=int(self.size_t.GetValue()),
                                                 size_c=int(self.size_c.GetValue()))
        