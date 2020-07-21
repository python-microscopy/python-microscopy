
import wx

class FocusLockPanel(wx.Panel):
    def __init__(self, parent, focus_PID, winid=-1, offset_piezo=None):
        """
        Parameters
        ----------
        offset_piezo : PYME.Acquire.Hardware.Piezos.offsetPiezoREST.OffsetPiezo
            offset piezo; only used to display current offset
        """
        wx.Panel.__init__(self, parent, winid)
        self.servo = focus_PID
        self.offset_piezo = offset_piezo

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.lock_checkbox = wx.CheckBox(self, -1, 'Lock')
        self.lock_checkbox.Bind(wx.EVT_CHECKBOX, self.OnToggleLock)
        hsizer.Add(self.lock_checkbox, 0, wx.ALL, 2)

        self.set_position_button = wx.Button(self, -1, 'Set Focus')
        hsizer.Add(self.set_position_button, 0, wx.ALL, 2)
        self.set_position_button.Bind(wx.EVT_BUTTON, self.OnUpdateSetpoint)

        self.set_subtraction_button = wx.Button(self, -1, 'Set Dark')
        hsizer.Add(self.set_subtraction_button, 0, wx.ALL, 2)
        self.set_subtraction_button.Bind(wx.EVT_BUTTON, self.OnSetSubtractionProfile)

        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)

        if self.offset_piezo is not None:
            pos = self.offset_piezo.GetPos()
            self._offset_slider = wx.Slider(self, -1, 100 * pos, 
                                           100 * self.offset_piezo.GetMin(), 
                                           100 * self.offset_piezo.GetMax(), 
                                           size=wx.Size(100, -1),
                                           style=wx.SL_HORIZONTAL)
            self._offset_label = wx.StaticBox(self, -1, u'%s - %2.3f %s' % ('offset', pos, u'\u03BCm'))
            
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(self.offset_slider, 0, wx.ALL, 2)
            hsizer.Add(self.offset_label, 0, wx.ALL, 2)
            sizer_1.Add(hsizer, 0, wx.EXPAND, 0)

        self.SetSizerAndFit(sizer_1)

    def OnToggleLock(self, event):
        self.servo.ToggleLock()

    def OnUpdateSetpoint(self, event):
        self.servo.ChangeSetpoint()

    def OnSetSubtractionProfile(self, event):
        self.servo.SetSubtractionProfile()

    def refresh(self):
        self.lock_checkbox.SetValue(bool(self.servo.lock_enabled))
        if self.offset_piezo is not None:
            pos = self.offset_piezo.GetOffset()

            self._offset_slider.SetValue(int(100 * pos))
            self._offset_slider.SetMin(100 * self.offset_piezo.GetMin())
            self._offset_slider.SetMax(100 * self.offset_piezo.GetMax())
            
            self._offset_label.SetLabel(u'%s - %2.3f %s' % ('offset', pos, u'\u03BCm'))
