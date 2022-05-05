
import wx

class MultiviewSelect(wx.Panel):
    def __init__(self, parent, scope, winid=-1, ):
        wx.Panel.__init__(self, parent, winid)
        self.scope = scope
        self.views = range(scope.cam.n_views)
        self.enabled_views = self.scope.cam.active_views

        sizer_1= wx.BoxSizer(wx.VERTICAL)
        l = wx.StaticText(self, -1, 'Select channels:')
        sizer_1.Add(l, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 2)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        self.buttons = []
        for view in self.views:
            sz = wx.BoxSizer(wx.HORIZONTAL)
            b = wx.ToggleButton(self, -1, str(view),size=(40, 25), style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_TOGGLEBUTTON, self.on_toggle)
            self.buttons.append(b)
            sz.Add(b, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
            sizer_2.Add(sz, 1, wx.EXPAND, 0)

        self.size_select_box = wx.ComboBox(self, -1, choices=['%d' % opt for opt in self.scope.cam.multiview_roi_size_options],
                                      value='%d' % self.scope.cam.size_x,  # for now, assume ROIs are symmetric
                                      size=(65, -1), style=wx.CB_DROPDOWN | wx.TE_PROCESS_ENTER)

        sizer_2.Add(self.size_select_box, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

        self.size_select_box.Bind(wx.EVT_COMBOBOX, self._on_combo_box)

        sizer_2.Fit(self)
        sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)

    def _on_combo_box(self, event):
        cb = event.GetEventObject()
        roi_size = int(cb.GetValue())

        # stop the frameWrangler, adjust size, and restart
        self.scope.frameWrangler.stop()
        self.scope.cam.ChangeMultiviewROISize(roi_size, roi_size)
        self.scope.frameWrangler.Prepare()
        self.scope.frameWrangler.start()

        # update the box in case the requested setting had to be altered
        wx.CallAfter(self.size_select_box.SetValue, '%d' % self.scope.cam.size_x)

    def on_toggle(self, event):
        """

        Parameters
        ----------
        event: wx.Event

        Returns
        -------

        """
        self.enabled_views = [view for view in self.views if self.buttons[view].GetValue()]
        self.scope.frameWrangler.stop()
        if (self.enabled_views is None) or (len(self.enabled_views) == 0):
            self.scope.cam.disable_multiview()
        else:
            self.scope.cam.enable_multiview(self.enabled_views)
        self.scope.frameWrangler.Prepare()
        self.scope.frameWrangler.start()


    def update(self):
        views = self.views
        for view in views:
            enabled=self.buttons[view].GetValue()
            if enabled:
                self.buttons[view].SetBackgroundColour("red")
            else:
                self.buttons[view].SetBackgroundColour(wx.NullColour)