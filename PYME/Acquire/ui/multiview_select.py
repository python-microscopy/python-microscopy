
import wx

class MultiviewSelect(wx.Panel):
    def __init__(self, parent, scope, winid=-1, ):
        wx.Panel.__init__(self, parent, winid)
        self.scope = scope
        self.enabled_views = []
        self.views=[0,1,2,3]
        self.buttons = []
        sizer_1= wx.BoxSizer(wx.VERTICAL)
        l = wx.StaticText(self, -1, 'Select channels:')
        sizer_1.Add(l, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        for view in self.views:
            sz = wx.BoxSizer(wx.HORIZONTAL)
            b = wx.ToggleButton(self, -1, str(view),size=(50,30), style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_TOGGLEBUTTON, self.on_toggle)
            self.buttons.append(b)
            sz.Add(b, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
            sizer_2.Add(sz, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND,0)
        self.SetSizer(sizer_1)
        self.SetSizer(sizer_1)

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