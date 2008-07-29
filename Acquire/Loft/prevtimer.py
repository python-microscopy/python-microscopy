import wx

class myTimer(wx.Timer):
    def Notify(self):
        if (cam.ExpReady()):
            cam.ExtractColor(ds.getCurrentChannelSlice(0), 1)
            cam.ExtractColor(ds.getCurrentChannelSlice(1), 2)
            cam.ExtractColor(ds.getCurrentChannelSlice(2), 3)
            cam.ExtractColor(ds.getCurrentChannelSlice(3), 4)
            fr.vp.imagepanel.Refresh()
