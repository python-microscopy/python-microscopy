import wx
from PYME.ui import cascading_layout


class TileSettingsUI(cascading_layout.CascadingLayoutPanel):
    def __init__(self, parent, scope, **kwargs):
        cascading_layout.CascadingLayoutPanel.__init__(self, parent, **kwargs)
        self.scope = scope
        
        if not hasattr(self.scope, 'tile_settings'):
            self.scope.tile_settings = {'n_tiles': (10, 10)}   

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, label='# x steps'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tXSteps = wx.TextCtrl(self, value='10')
        self.tXSteps.Bind(wx.EVT_TEXT, self.update_settings)
        hsizer.Add(self.tXSteps, 0, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, label='# y steps'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tYSteps = wx.TextCtrl(self, value='10')
        self.tYSteps.Bind(wx.EVT_TEXT, self.update_settings)
        hsizer.Add(self.tYSteps, 0, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND)

        self.update_from_settings()

        self.SetSizerAndFit(vsizer)

    def update_from_settings(self):
        xs, ys = self.scope.tile_settings.get('n_tiles', (10, 10))
        self.tXSteps.SetValue(str(xs))
        self.tYSteps.SetValue(str(ys))

    def update_settings(self, event=None):
        self.scope.tile_settings['n_tiles'] = (int(self.tXSteps.GetValue()), int(self.tYSteps.GetValue()))

