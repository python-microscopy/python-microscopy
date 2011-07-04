import wx
import os

class DeconvSettingsDialog(wx.Dialog):
    def __init__(self, parent, beadMode=False):
        wx.Dialog.__init__(self, parent, title='Deconvolution')

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        notebook = wx.Notebook(self, -1)

        #basic panel
        pan1 = wx.Panel(notebook, -1)
        notebook.AddPage(pan1, 'Basic')

        sizer2 = wx.BoxSizer(wx.VERTICAL)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)

        if not beadMode:
            sizer3.Add(wx.StaticText(pan1, -1, 'PSF:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.fpPSF = wx.FilePickerCtrl(pan1, -1, wildcard='*.psf', style=wx.FLP_OPEN|wx.FLP_FILE_MUST_EXIST)
            self.fpPSF.Bind(wx.EVT_FILEPICKER_CHANGED, self.OnPSFFileChanged)

            sizer3.Add(self.fpPSF, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        else:
            sizer3.Add(wx.StaticText(pan1, -1, 'Bead Diameter [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            self.tBeadSize = wx.TextCtrl(pan1, -1, '200')

            sizer3.Add(self.tBeadSize, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Method:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.cMethod = wx.Choice(pan1, -1, choices=['ICTM', 'Richardson-Lucy'])

        sizer3.Add(self.cMethod, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Number of iterations:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tNumIters = wx.TextCtrl(pan1, -1, '10')

        sizer3.Add(self.tNumIters, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)
        
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Offset:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tOffset = wx.TextCtrl(pan1, -1, '0')

        sizer3.Add(self.tOffset, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, u'Regularisation \u03BB:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tRegLambda = wx.TextCtrl(pan1, -1, '1e-1')

        sizer3.Add(self.tRegLambda, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        pan1.SetSizerAndFit(sizer2)

        #blocking panel
        pan1 = wx.Panel(notebook, -1)
        notebook.AddPage(pan1, 'Blocking')

        sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.cbBlocking = wx.CheckBox(pan1, -1, 'Do tiled/blocked deconvolution')
        sizer2.Add(self.cbBlocking, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3.Add(wx.StaticText(pan1, -1, 'Tile size:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tTileSize = wx.TextCtrl(pan1, -1, '128')

        sizer3.Add(self.tTileSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2.Add(sizer3, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        pan1.SetSizerAndFit(sizer2)



        sizer1.Add(notebook, 1, wx.EXPAND|wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()

        self.bOK = wx.Button(self, wx.ID_OK)
        if not beadMode:
            self.bOK.Disable()
        self.bOK.SetDefault()

        btSizer.AddButton(self.bOK)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def GetNumIterationss(self):
        return int(self.tNumIters.GetValue())

    def GetRegularisationLambda(self):
        return float(self.tRegLambda.GetValue())

    def GetMethod(self):
        return self.cMethod.GetStringSelection()

    def GetBlocking(self):
        return self.cbBlocking.GetValue()

    def GetBlockSize(self):
        return int(self.tTileSize.GetValue())

    def GetBeadRadius(self):
        return float(self.tBeadSize.GetValue())/2

    def GetPSFFilename(self):
        return self.fpPSF.GetPath()

    def OnPSFFileChanged(self, event):
        self.bOK.Enable(os.path.exists(self.fpPSF.GetPath()))
        
    def GetOffset(self):
        return float(self.tOffset.GetValue())


class DeconvProgressDialog(wx.Dialog):
    def __init__(self, parent, numIters):
        wx.Dialog.__init__(self, parent, title='Deconvolution Progress')
        self.cancelled = False

        self.numIters = numIters

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.gProgress = wx.Gauge(self, -1, numIters)

        sizer1.Add(self.gProgress, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_CANCEL)
        btn.Bind(wx.EVT_BUTTON, self.OnCancel)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnCancel(self, event):
        self.cancelled = True
        #self.EndModal(wx.ID_CANCEL)

    def Tick(self, dec):
        if not self.cancelled:
            self.gProgress.SetValue(dec.loopcount)
            return True
        else:
            return False

class DeconvProgressPanel(wx.Panel):
    def __init__(self, parent, numIters):
        wx.Panel.__init__(self, parent)
        self.cancelled = False

        self.numIters = numIters

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)

        self.gProgress = wx.Gauge(self, -1, numIters)

        sizer1.Add(self.gProgress, 5, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        #btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_CANCEL)
        btn.Bind(wx.EVT_BUTTON, self.OnCancel)

        #btSizer.AddButton(btn)

        #btSizer.Realize()

        sizer1.Add(btn, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnCancel(self, event):
        self.cancelled = True
        #self.EndModal(wx.ID_CANCEL)

    def Tick(self, dec):
        if not self.cancelled:
            self.gProgress.SetValue(dec.loopcount)
            return True
        else:
            return False

    

    
   