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

        #vsizer = wx.BoxSizer(wx.VERTICAL)

        self.vp = vp

        #self.ds = vp.ds
        self.dsa = cSMI.CDataStack_AsArray(self.vp.ds, 0)[:,:,0].ravel()
        
        #self.do = vp.do
        

        self.scale = 2

        self.vp.do.Optimise(self.vp.ds)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.hlDispMapping = histLimits.HistLimitPanel(self, -1, self.dsa, self.vp.do.getDisp1Off(), self.vp.do.getDisp1Off() + 255./self.vp.do.getDisp1Gain(), True, size=(100, 80))
        self.hlDispMapping.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnMappingChange)

        hsizer.Add(self.hlDispMapping, 1, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        vsizer2 = wx.BoxSizer(wx.VERTICAL)
        self.bOptimise = wx.Button(self, -1, 'Optimise', style=wx.BU_EXACTFIT)
        self.bOptimise.Bind(wx.EVT_BUTTON, self.OnBOptimise)
        vsizer2.Add(self.bOptimise, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)


        self.cbAutoOptimise = wx.CheckBox(self, -1, 'Auto')
        #self.cbAutoOptimise.Bind(wx.EVT_CHECKBOX, self.OnCbAutoOpt)
        vsizer2.Add(self.cbAutoOptimise, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 3)
        vsizer2.Add((0,0), 1, 0, 0)

        #hsizer.Add(vsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        #vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)
        
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)

        #scaleSizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, "Scale"), wx.HORIZONTAL)
        self.cbScale = wx.ComboBox(self, -1, choices=["1:4", "1:2", "1:1", "2:1", "4:1"], style=wx.CB_DROPDOWN|wx.CB_READONLY, size=(55, -1))
        self.cbScale.Bind(wx.EVT_COMBOBOX, self.OnScaleChanged)
        self.cbScale.SetSelection(self.scale)

        #scaleSizer.Add(self.cbScale, 0, wx.ALL, 5)
        #hsizer.Add(wx.StaticText(self, -1, 'Scale: '), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        vsizer2.Add(self.cbScale, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 5)

        hsizer.Add(vsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        #vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)

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

