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
        hsizer.Add(wx.StaticText(self, label='# steps - x:'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tXSteps = wx.TextCtrl(self, value='10', size=(40, -1))
        self.tXSteps.Bind(wx.EVT_TEXT, self.update_settings)
        hsizer.Add(self.tXSteps, 1, wx.ALIGN_CENTER_VERTICAL)
        #vsizer.Add(hsizer, 0, wx.EXPAND)

        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #hsizer.Add(wx.StaticText(self, label='# y steps'), 0, wx.ALIGN_CENTER_VERTICAL)
        hsizer.Add(wx.StaticText(self, label=', y:'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tYSteps = wx.TextCtrl(self, value='10', size=(40, -1))
        self.tYSteps.Bind(wx.EVT_TEXT, self.update_settings)
        hsizer.Add(self.tYSteps, 1, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, label='Tile spacing:'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tTileSpacing = wx.TextCtrl(self, value='0.8', size=(40, -1))
        self.tTileSpacing.Bind(wx.EVT_TEXT, self.update_settings)
        self.tTileSpacing.SetToolTip('Spacing between tiles as a fraction of the tile size')
        hsizer.Add(self.tTileSpacing, 1, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 2)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stRegionSize = wx.StaticText(self, label='Region size: 0.0 x 0.0 um')
        hsizer.Add(self.stRegionSize, 0, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #hsizer.Add(wx.StaticText(self, label=''), 0, wx.ALIGN_CENTER_VERTICAL)
        self.cbSaveRaw = wx.CheckBox(self, label='Save raw frames')
        self.cbSaveRaw.SetValue(False)
        self.cbSaveRaw.SetToolTip('Save raw frames in addition to the stitched image. Can be useful for debugging and/or high-precision alignment')
        self.cbSaveRaw.Bind(wx.EVT_CHECKBOX, self.update_settings)
        hsizer.Add(self.cbSaveRaw, 0, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND|wx.TOP, 5)

        self.update_from_settings()
        self.update_region_size()

        self.SetSizerAndFit(vsizer)

    def update_from_settings(self):
        xs, ys = self.scope.tile_settings.get('n_tiles', (10, 10))
        self.tXSteps.SetValue(str(xs))
        self.tYSteps.SetValue(str(ys))
        self.tTileSpacing.SetValue('%3.2f' %self.scope.tile_settings.get('tile_spacing', 0.8))

    def update_settings(self, event=None):
        self.scope.tile_settings['n_tiles'] = (int(self.tXSteps.GetValue()), int(self.tYSteps.GetValue()))
        self.scope.tile_settings['tile_spacing'] = float(self.tTileSpacing.GetValue())
        self.scope.tile_settings['save_raw'] = self.cbSaveRaw.GetValue()

        self.update_region_size()

    def update_region_size(self):
        import numpy as np
        n = np.array(self.scope.tile_settings.get('n_tiles', (10, 10)))
        sp = self.scope.tile_settings.get('tile_spacing', 0.8)
        fs = np.array(self.scope.frameWrangler.currentFrame.shape[:2])*np.array(self.scope.GetPixelSize())
        #print(((n-1) * sp * fs + fs))
        self.stRegionSize.SetLabel('Region size: %3.1f x %3.1f um' % tuple(n * sp * fs + fs))
        

class ZTileSettingsUI(cascading_layout.CascadingLayoutPanel):
    def __init__(self, parent, scope, **kwargs):
        from PYME.Acquire.ui.seqdialog import seqPanel
        cascading_layout.CascadingLayoutPanel.__init__(self, parent, **kwargs)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)

        z_panel = seqPanel(self, scope, mode='compact')
        vsizer.Add(z_panel, 0, wx.EXPAND)

        tile_panel = TileSettingsUI(self, scope)
        vsizer.Add(tile_panel, 0, wx.EXPAND)

        self.SetSizerAndFit(vsizer)