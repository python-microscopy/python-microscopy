
import wx


class CameraMapDialog(wx.Dialog):
    def __init__(self, parent, scope):
        wx.Dialog.__init__(self, parent, -1, 'Camera Calibrations')
        self.scope = scope

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.cam_names = list(scope.cameras.keys())
        self.cal_choices = []

        gsizer = wx.FlexGridSizer(2, 2, 2)
        gsizer.AddGrowableCol(1, 1)
        for name in self.cam_names:
            gsizer.Add(wx.StaticText(self, -1, '%s: ' % name), 0, wx.ALIGN_CENTER_VERTICAL, 0)
            ch = wx.Choice(self, -1)
            ch.Bind(wx.EVT_CHOICE, self.OnChooseMaps)
            gsizer.Add(ch, 1, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
            self.cal_choices.append(ch)

        self._set_calibration_choices()

        sizer1.Add(gsizer, 0, wx.EXPAND | wx.ALL, 5)

        hsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'New Setting'), wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Name: '), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self._name = wx.TextCtrl(self, -1, size=(80, -1))
        hsizer.Add(self._name, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        hsizer.Add(wx.StaticText(self, -1, 'dark path:'), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self._dark = wx.TextCtrl(self, -1, size=(30, -1))
        hsizer.Add(self._dark, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        hsizer.Add(wx.StaticText(self, -1, 'flat path:'), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self._flat = wx.TextCtrl(self, -1, size=(30, -1))
        hsizer.Add(self._flat, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        hsizer.Add(wx.StaticText(self, -1, 'variance path:'), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self._variance = wx.TextCtrl(self, -1, size=(30, -1))
        hsizer.Add(self._variance, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        # hsizer.Add(wx.StaticText(self, -1, ') [\u03BCm] '), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        self.bAdd = wx.Button(self, -1, 'Add', style=wx.BU_EXACTFIT)
        self.bAdd.Bind(wx.EVT_BUTTON, self.OnAddMaps)
        hsizer.Add(self.bAdd, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        sizer1.Add(hsizer, 0, wx.EXPAND | wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        # btn = wx.Button(self, wx.ID_CANCEL)

        # btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnAddMaps(self, event):
        self.scope.AddCameraMaps(self._name.GetValue(), self._dark.GetValue(), self._flat.GetValue(), self._variance.GetValue())
        self._set_calibration_choices()

    def OnChooseMaps(self, event):
        choice = event.GetEventObject()
        cam_name = self.cam_names[self.cal_choices.index(choice)]
        cal_name = self.cal_choices[choice.GetStringSelection()]

        self.scope.SetCameraMaps(cal_name, cam_name)

    def _set_calibration_choices(self):
        camera_maps = self.scope.settingsDB.execute("SELECT ID, name, dark_path, flat_path, var_path FROM CameraMaps ORDER BY ID DESC").fetchall()

        map_ids = []
        self.map_names = {}

        for ch in self.cal_choices:
            ch.Clear()

        for ID, name, dark_path, flat_path, var_path in camera_maps:
            comp_name = '%s - (%s, %s, %s)' % (name, dark_path, flat_path, var_path)

            map_ids.append(ID)
            self.map_names[comp_name] = name

            for ch in self.cal_choices:
                ch.Append(comp_name)

        for ch, cam_name in zip(self.cal_choices, self.cam_names):
            try:
                curr_choice_id = self.scope.settingsDB.execute(
                    "SELECT choice_id FROM CameraMapsHistory WHERE cam_serial=? ORDER BY time DESC",
                    (self.scope.cameras[cam_name].GetSerialNumber(),)).fetchone()[0]

                if curr_choice_id is not None:
                    ch.SetSelection(map_ids.index(curr_choice_id))
            except:
                pass
