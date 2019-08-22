
import wx

class FocusLockPanel(wx.Panel):
    def __init__(self, parent, focus_PID, winid=-1):
        wx.Panel.__init__(self, parent, winid)
        self.servo = focus_PID

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

        self.SetSizerAndFit(sizer_1)

    def OnToggleLock(self, event):
        self.servo.ToggleLock()

    def OnUpdateSetpoint(self, event):
        self.servo.ChangeSetpoint()

    def OnSetSubtractionProfile(self, event):
        self.servo.SetSubtractionProfile()

    def refresh(self):
        self.lock_checkbox.SetValue(bool(self.servo.lock_enabled))
