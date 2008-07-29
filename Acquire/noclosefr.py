import wx

class noCloseFrame(wx.Frame):
    def __init__(self,*args, **kwds):
        wx.Frame.__init__(self,*args, **kwds)
        wx.EVT_CLOSE(self, self.OnCloseWindow)

    def OnCloseWindow(self, event):   
        if (not event.CanVeto()): 
            self.Destroy()
        else:
            event.Veto()
            self.Hide()
            
class wxFrame(wx.Frame):
    def __init__(self,*args, **kwds):
        wx.Frame.__init__(self,*args, **kwds)
        wx.EVT_CLOSE(self, self.OnCloseWindow)

    def OnCloseWindow(self, event):   
        if (not event.CanVeto()): 
            self.Destroy()
        else:
            event.Veto()
            self.Hide()