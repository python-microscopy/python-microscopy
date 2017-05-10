# -*- coding: utf-8 -*-

import wx
import wx.grid
import numpy as np

class EditGrid(wx.grid.Grid):
    def __init__(self, *args, **kwargs):
        wx.grid.Grid.__init__(self, *args, **kwargs)
        wx.EVT_KEY_DOWN(self, self.OnKeyDown)
        
    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        
        if event.ControlDown and key == ord('V'):
            self.OnPaste(event)
        else:
            event.Skip()
        
    def toarray(self, selection=None):
        if selection:
            x0, y0, x1, y1 = selection
        else:
            x0, y0, x1, y1 = self._getvalidbounds()
            
        out = np.zeros([x1-x0, y1-y0], 'd')
        
        for i in range(x0, x1):
            for j in range(y0, y1):
                out[i,j]= float(self.GetCellValue(i,j))
                
        return out
            
            
    def _getvalidbounds(self):
        x0 = 0
        y0 = 0
        
        x1 = 0
        y1 = 0
        
        while y1 <= self.GetNumberCols() and not self.GetCellValue(0, y1) == '':
            y1 += 1
            
        while x1 <= self.GetNumberRows() and not self.GetCellValue(x1, 0) =='':
            x1 += 1
            
        return x0, y0, x1, y1
        
        
    def setarray(self, data,x0=0, y0=0):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.SetCellValue(i+x0, j+y0, '%s' % data[i, j])
                
    
    def tostring(self, selection=None):
        from io import BytesIO
        sb = BytesIO()
        
        np.savetxt(sb, self.toarray(selection), delimiter='\t')
        
        return sb.getvalue()
        
    def setfromstring(self, data, x0=0, y0=0):
        from io import BytesIO
        #print repr(data)
        sb = BytesIO(data.encode())
        
        self.setarray(np.loadtxt(sb, delimiter = '\t'), x0, y0)
        
    def OnPaste(self, event):
        cb = wx.TextDataObject()
        
        wx.TheClipboard.Open()
        wx.TheClipboard.GetData(cb)
        wx.TheClipboard.Close()
        
        self.setfromstring(cb.GetText())
        
        
class EntryGrid(wx.Frame):
    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, size=(500, 500))
        
        self.grid = EditGrid(self)
        
        self.grid.CreateGrid(100, 5)
        
    @property
    def data(self):
        return self.grid.toarray()
        
def ShowDataGrid():
    f = EntryGrid()
    f.Show()
    return f
        
        
        
        