import wx
from PYME.Acquire.Hardware.DigiData.DigiDataClient import getDDClient

dd = getDDClient()

class MyFrame(wx.MiniFrame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self, -1)

        v2sizer = wx.BoxSizer(wx.VERTICAL)

        self.stTemp = wx.StaticText(panel, -1, '25.00 C')
        self.stTemp.SetFont(wx.Font(30, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        v2sizer.Add(self.stTemp)
        panel.SetSizerAndFit(v2sizer)

        vsizer.Add(panel)

        self.SetSizerAndFit(vsizer)

        self.onTimer()

        self.timer = wx.Timer(self, -1)
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.timer.Start(2000)



    def onTimer(self, ev = None):
        temp = dd.GetAIValue(1)*1000./2.**15 - 273.15

        self.stTemp.SetLabel('%3.2f C' % temp)


if __name__ == "__main__":
    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    fTaskMon = MyFrame(None, -1, "Temperature")
    app.SetTopWindow(fTaskMon)
    fTaskMon.Show()
    app.MainLoop()

