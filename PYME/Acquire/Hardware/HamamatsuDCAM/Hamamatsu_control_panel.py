import wx

# Mostly copied from PYME.Acquire.Hardware.pco.pco_sdk_cam_control_panel
class ModeControl(wx.Panel):
    def __init__(self, parent, cam):
        wx.Panel.__init__(self, parent)
        self.scope = parent.scope
        self.parent = parent
        self.cam = cam
        self.options = ["Single shot", "Continuous", "Software trigger", "Hardware trigger"]
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, "Mode : "), 1, wx.ALL, 2)
        
        self.choice = wx.Choice(self, -1, size = [100,-1], choices=self.options)
        self.choice.Bind(wx.EVT_CHOICE, self.on_change)
        hsizer.Add(self.choice, 0, wx.ALL, 2)
        
        self.update()
        
        self.SetSizerAndFit(hsizer)
        
    def on_change(self, event=None):
        self.scope.frameWrangler.stop()
        self.cam.SetAcquisitionMode(int(self.choice.GetSelection()))
        self.scope.frameWrangler.start()
        self.parent.update()
        
    def update(self):
        self.choice.SetSelection(self.cam.GetAcquisitionMode())

class HamamatsuControl(wx.Panel):
    def __init__(self, parent, cam, scope):
        wx.Panel.__init__(self, parent)
        
        self.cam = cam
        self.scope = scope
        
        self.ctrls = [ModeControl(self, cam)]
        
        self._init_ctrls()

    def _init_ctrls(self):
        vsizer = wx.BoxSizer(wx.VERTICAL)

        for c in self.ctrls:
            vsizer.Add(c, 0, wx.EXPAND|wx.ALL, 2)
        
        self.SetSizerAndFit(vsizer)
        
    def update(self):
        for c in self.ctrls:
            c.update()
            