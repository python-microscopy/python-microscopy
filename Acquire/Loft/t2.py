import wx
import example

class myFrame(wx.Frame):
    """
    This is MyFrame.  It just shows a few controls on a wxPanel,
    and has a simple menu.
    """
    def __init__(self, parent, title, dstack = None):
        wx.Frame.__init__(self, parent, -1, title,
                          pos=(150, 150), size=(350, 200))
        
        if (dstack == None):
            self.ds = example.CDataStack("test.kdf")
        else:
            self.ds = dstack

        self.SetSize(wx.Size(self.ds.getWidth(),self.ds.getHeight()))

        self.im = wx.EmptyImage(self.ds.getWidth(),self.ds.getHeight())

        self.do = example.CDisplayOpts()
        self.do.setDisp1Chan(0)
        self.do.setDisp2Chan(0)
        self.do.setDisp3Chan(0)

        self.do.setDisp1Gain(1)
        self.do.setDisp2Gain(1)
        self.do.setDisp3Gain(1)

        self.rend = example.CLUT_RGBRenderer()
        self.rend.setDispOpts(self.do)

        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_KEY_DOWN(self, self.OnKeyPress)

        


    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)
        #self.PrepareDC(dc)
        #dc.BeginDrawing()
        #mdc = wx.MemoryDC(dc)

        bmp = self.im.GetDataBuffer()
        self.rend.pyRender(bmp,self.ds)

        dc.DrawBitmap(wx.BitmapFromImage(self.im),wx.Point(0,0))
        #mdc.SelectObject(wx.BitmapFromImage(self.im))
        #mdc.DrawBitmap(wx.BitmapFromImage(self.im),wx.Point(0,0))
        #dc.Blit(0,0,im.GetWidth(), im.GetHeight(),mdc,0,0)
        #dc.EndDrawing()

    def OnWheel(self, event):
        rot = event.GetWheelRotation()
        if rot < 0:
            self.ds.setZPos(self.ds.getZPos() - 1)

        if rot > 0:
            self.ds.setZPos(self.ds.getZPos() + 1)

        self.Refresh()
    
    def OnKeyPress(self, event):
        if event.GetKeyCode() == wx.WXK_PRIOR:
            self.ds.setZPos(self.ds.getZPos() - 1)
            self.Refresh()
        elif event.GetKeyCode() == wx.WXK_NEXT:
            self.ds.setZPos(self.ds.getZPos() + 1)
            self.Refresh()
        else:
            event.Skip()

        