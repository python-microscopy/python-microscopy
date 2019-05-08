import wx

class ScannerPanel(wx.Panel):
    def __init__(self, parent, scanner):
        wx.Panel.__init__(self, parent)

        self.scanner = scanner

        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Scan Frequency:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.tScanFreq = wx.TextCtrl(self, -1, '500', style=wx.TE_PROCESS_ENTER)
        hsizer.Add(self.tScanFreq, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Scan Amplitude:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.tScanAmp = wx.TextCtrl(self, -1, '0.03', style=wx.TE_PROCESS_ENTER)
        hsizer.Add(self.tScanAmp, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cbEnableScan = wx.CheckBox(self, -1, 'Enable Scanner')
        hsizer.Add(self.cbEnableScan, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.SetSizerAndFit(vsizer)

        self.cbEnableScan.Bind(wx.EVT_CHECKBOX, self.OnToggleScan)
        self.tScanAmp.Bind(wx.EVT_TEXT_ENTER, self.OnToggleScan)
        self.tScanFreq.Bind(wx.EVT_TEXT_ENTER, self.OnToggleScan)


    def OnToggleScan(self, event=None):
        if self.cbEnableScan.GetValue():
            self.scanner.start_scanning(frequency=float(self.tScanFreq.GetValue()),
                                        amplitude=float(self.tScanAmp.GetValue()),
                                        waveform='tri')
        else:
            self.scanner.stop_scanning()
