import wx

class SplashPanel(wx.Panel):
    def __init__(self, parent, size=(-1,-1)):
        wx.Panel.__init__(self, parent, size = size)

        self.tickcount = 0
        self.messages = {}
        
        wx.EVT_PAINT(self, self.OnPaint)
        
    def DoPaint(self, dc):
        dc.Clear()

        yp = 10
        
        dc.SetFont(wx.Font(20, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(wx.Colour(57, 76, 135))
        
        ts = dc.GetTextExtent('PYME Acquire')
        
        dc.DrawText('PYME Acquire', self.Size[0]/2 - ts[0]/2, yp)

        yp += ts[1]
        
        dc.SetFont(wx.NORMAL_FONT)
        dc.SetTextForeground(wx.BLACK)

        yp += 10

        ts = dc.GetTextExtent('Initialising ...')

        dc.DrawText('Initialising .' + ''.join([' .' for i in range(self.tickcount)]), 20, yp)
        yp += ts[1]
        yp += 10

        for msg in self.messages.values():
            dc.DrawText(msg, 30, yp)
            yp += ts[1]
        

        dc.SetTextForeground(wx.BLACK)

        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
        dc.SetFont(wx.NullFont)

    def OnPaint(self,event):
        DC = wx.PaintDC(self)
        self.PrepareDC(DC)

        s = self.GetVirtualSize()
        MemBitmap = wx.EmptyBitmap(s.GetWidth(), s.GetHeight())
        #del DC
        MemDC = wx.MemoryDC()
        OldBitmap = MemDC.SelectObject(MemBitmap)
        try:
            DC.BeginDrawing()

            self.DoPaint(MemDC);

            DC.Blit(0, 0, s.GetWidth(), s.GetHeight(), MemDC, 0, 0)
            DC.EndDrawing()
        finally:

            del MemDC
            del MemBitmap

class SplashScreen(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "window title", pos=(300, 300),
                         style=wx.SUNKEN_BORDER|wx.STAY_ON_TOP)
        self.SetSize((400, 300))

        self.panel = SplashPanel(self, size=self.GetSize())
        self.parent = parent

    def Tick(self):
        self.panel.tickcount +=1
        self.panel.Refresh()
        if self.parent.initDone:
            self.parent.time1.WantNotification.remove(self.Tick)
            self.Destroy()

    def SetMessage(self, key, msg):
        self.panel.messages[key] = msg
        

