import wx
from PYME.Analysis.LMVis import histLimits
from PYME import cSMI

class dispSettingsFrame(wx.Frame):
    def __init__(self, parent, vp):
        wx.Frame.__init__(self, parent, title='Display')

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.dispPanel = dispSettingsPanel(self, vp)

        hsizer.Add(self.dispPanel, 0, wx.EXPAND|wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)


class dispSettingsPanel(wx.Panel):
    def __init__(self, parent, vp):
        wx.Panel.__init__(self, parent)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.vp = vp

        #self.ds = vp.ds
        self.dsa = cSMI.CDataStack_AsArray(self.vp.ds, 0)[:,:,0].ravel()
        
        #self.do = vp.do
        

        self.scale = 2

        self.vp.do.Optimise(self.vp.ds)

        self.hlDispMapping = histLimits.HistLimitPanel(self, -1, self.dsa, self.vp.do.getDisp1Off(), self.vp.do.getDisp1Off() + 255./self.vp.do.getDisp1Gain(), True, size=(200, 100))
        self.hlDispMapping.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnMappingChange)

        vsizer.Add(self.hlDispMapping, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bOptimise = wx.Button(self, -1, 'Optimise')
        self.bOptimise.Bind(wx.EVT_BUTTON, self.OnBOptimise)
        hsizer.Add(self.bOptimise, 0, wx.ALIGN_CENTER_VERTICAL|wx.TOP|wx.BOTTOM|wx.RIGHT, 5)

        self.cbAutoOptimise = wx.CheckBox(self, -1, 'Auto')
        #self.cbAutoOptimise.Bind(wx.EVT_CHECKBOX, self.OnCbAutoOpt)
        hsizer.Add(self.cbAutoOptimise, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        

        scaleSizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, "Scale"), wx.HORIZONTAL)
        self.cbScale = wx.ComboBox(self, -1, choices=["1:4", "1:2", "1:1", "2:1", "4:1"], style=wx.CB_DROPDOWN|wx.CB_READONLY, size=(80, -1))
        self.cbScale.Bind(wx.EVT_COMBOBOX, self.OnScaleChanged)
        self.cbScale.SetSelection(self.scale)

        scaleSizer.Add(self.cbScale, 0, wx.ALL, 5)

        hsizer.Add(scaleSizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        self.SetSizer(vsizer)
        vsizer.Fit(self)

    def OnMappingChange(self, event):
        lower, upper = self.hlDispMapping.GetValue()

        off = 1.*(lower)
        gain = 255./(upper - lower)

        self.vp.do.setDisp3Gain(gain)
        self.vp.do.setDisp2Gain(gain)
        self.vp.do.setDisp1Gain(gain)

        self.vp.do.setDisp3Off(off)
        self.vp.do.setDisp2Off(off)
        self.vp.do.setDisp1Off(off)

        self.vp.Refresh()

    def OnScaleChanged(self, event):
        self.scale = self.cbScale.GetSelection()

        self.vp.SetScale(self.scale)

    def OnBOptimise(self, event):
        self.vp.do.Optimise(self.vp.ds)

        self.hlDispMapping.SetValue((self.vp.do.getDisp1Off(), self.vp.do.getDisp1Off() + 255./self.vp.do.getDisp1Gain()))
        self.vp.Refresh()

    #def OnCBAutoOpt(self, event):

    def RefrData(self, caller=None):
        #if self.hlDispMapping.dragging == None:
        self.dsa = cSMI.CDataStack_AsArray(self.vp.ds, 0)[:,:,0].ravel()
        self.hlDispMapping.SetData(self.dsa, self.hlDispMapping.limit_lower, self.hlDispMapping.limit_upper)

        if self.cbAutoOptimise.GetValue():
            self.OnBOptimise(None)

