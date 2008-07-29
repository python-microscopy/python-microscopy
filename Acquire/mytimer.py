import wx

class mytimer(wx.Timer):
    def __init__(self):
        wx.Timer.__init__(self)
        self.WantNotification = []

    def Notify(self):
        for a in self.WantNotification:
                a()