
import wx

class FocusLockPanel(wx.Panel):
    def __init__(self, parent, focus_PID, winid=-1):
        wx.Panel.__init__(self, parent, winid)
        self.servo = focus_PID

        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        # self.cbTrack = wx.CheckBox(self, -1, 'Track')
        # hsizer.Add(self.cbTrack, 0, wx.ALL, 2)
        # self.cbTrack.Bind(wx.EVT_CHECKBOX, self.OnCBTrack)
        self.lock_checkbox = wx.CheckBox(self, -1, 'Lock')
        self.lock_checkbox.Bind(wx.EVT_CHECKBOX, self.OnToggleLock)
        hsizer.Add(self.lock_checkbox, 0, wx.ALL, 2)
        # self.cbLockActive = wx.CheckBox(self, -1, 'Lock Active')
        # self.cbLockActive.Enable(False)
        # hsizer.Add(self.cbLockActive, 0, wx.ALL, 2)
        # sizer_1.Add(hsizer, 0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.set_position_button = wx.Button(self, -1, 'Set focus to current')
        hsizer.Add(self.set_position_button, 0, wx.ALL, 2)
        self.set_position_button.Bind(wx.EVT_BUTTON, self.OnUpdateSetpoint)

        sizer_1.Add(hsizer, 0, wx.EXPAND, 0)

        # hsizer = wx.BoxSizer(wx.HORIZONTAL)
        # hsizer.Add(wx.StaticText(self, -1, "Tolerance [nm]:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        # self.tTolerance = wx.TextCtrl(self, -1, '%3.0f' % (1e3 * self.dt.get_focus_tolerance()), size=[30, -1])
        # hsizer.Add(self.tTolerance, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetTolerance = wx.Button(self, -1, 'Set', style=wx.BU_EXACTFIT)
        # hsizer.Add(self.bSetTolerance, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        # self.bSetTolerance.Bind(wx.EVT_BUTTON, self.OnBSetTolerance)
        # sizer_1.Add(hsizer, 0, wx.EXPAND, 0)

    def OnToggleLock(self, event):
        self.servo.ToggleLock(self.lock_checkbox.GetValue())

    def OnUpdateSetpoint(self, event):
        self.servo.ChangeSetpoint()

    def refresh(self):
        # try:
            # calibState, NStates = self.dt.get_calibration_state()
            # self.gCalib.SetRange(NStates + 1)
            # self.gCalib.SetValue(calibState)
            #
            # try:
            #     t, dx, dy, dz, corr, corrmax, poffset, pos = self.dt.get_history(1)[-1]
            #     self.stError.SetLabel(("Error: x = %s nm y = %s nm\n" +
            #                            "z = %s nm noffs = %s nm c/cm = %4.2f") %
            #                           ("{:>+3.2f}".format(dx), "{:>+3.2f}".format(dy),
            #                            "{:>+6.1f}".format(1e3 * dz), "{:>+6.1f}".format(1e3 * poffset),
            #                            corr / corrmax))
            #
            # except IndexError:
            #     pass

        self.lock_checkbox.SetValue(self.servo.lock_enabled)

        #     if (len(self.dt.get_history(0)) > 0) and (
        #             len(self.dt.get_history(0)) % self.plotInterval == 0) and self.showPlots:
        #         self.trackPlot.draw()
        # except AttributeError:
        #     logger.exception('error in refresh')
        #     pass
        # except IndexError:
        #     logger.exception('error in refresh')
        #     pass