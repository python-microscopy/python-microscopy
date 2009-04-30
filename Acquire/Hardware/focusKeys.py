import wx

class FocusKeys:
    def __init__(self, parent, menu, piezo, keys = ['F1', 'F2', 'F3', 'F4']):
        self.piezo = piezo
        self.focusIncrement = 0.2

        idFocUp = wx.NewId()
        idFocDown = wx.NewId()
        idSensUp = wx.NewId()
        idSensDown = wx.NewId()

        self.menu = wx.Menu(title = '')

        self.menu.Append(idFocDown, 'Focus Down\t%s' % keys[0])
        wx.EVT_MENU(parent, idFocDown, self.OnFocDown)

        self.menu.Append(idFocUp, 'Focus Up\t%s' % keys[1])
        wx.EVT_MENU(parent, idFocUp, self.OnFocUp)

        self.menu.Append(idSensDown, 'Sensitivity Down\t%s' % keys[2])
        wx.EVT_MENU(parent, idSensDown, self.OnSensDown)

        self.menu.Append(idSensUp, 'Sensitivity Up\t%s' % keys[3])
        wx.EVT_MENU(parent, idSensUp, self.OnSensUp)

        menu.Append(menu=self.menu, title = 'Focus')
        self.mbar = menu
        self.mpos = menu.GetMenuCount() - 1


    def OnFocDown(self,event):
        #p = self.piezo[0].GetPos(self.piezo[1])
        p = self.piezo[0].lastPos
        self.piezo[0].MoveTo(self.piezo[1], p - self.focusIncrement)

    def OnFocUp(self,event):
        #p = self.piezo[0].GetPos(self.piezo[1])
        p = self.piezo[0].lastPos
        self.piezo[0].MoveTo(self.piezo[1], p + self.focusIncrement)

    def OnSensDown(self,event):
        if self.focusIncrement > 0.05:
            self.focusIncrement /= 2.

    def OnSensUp(self,event):
        if self.focusIncrement < 10:
            self.focusIncrement *= 2.

    def refresh(self):
        self.mbar.SetMenuLabel(self.mpos, 'Focus = %3.2f, Inc = %3.2f' %(self.piezo[0].GetPos(self.piezo[1]), self.focusIncrement))


        

        
