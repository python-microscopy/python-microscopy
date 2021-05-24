import wx
from ._base import Plugin

class RedimensionDialog(wx.Dialog):
    def __init__(self, parent, img):
        wx.Dialog.__init__(self, parent, title='Redimension')
        self._img = img
        
        _, _, sz, st, sc = img.data_xyztc.shape
        
        self._dim_prod = sz*st*sc
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        gsizer = wx.GridSizer(2, gap=(5,5))
        
        gsizer.Add(wx.StaticText(self, -1, 'Input axis order:'))
        self.cOrder = wx.Choice(self, -1, choices=['XYZTC', 'XYCZT', 'XYZCT', 'XYCTZ', 'XYTZC', 'XYTCZ'])
        gsizer.Add(self.cOrder)
        
        gsizer.Add(wx.StaticText(self, -1, 'Size Z:'))
        self.tSizeZ = wx.TextCtrl(self, -1, str(sz))
        gsizer.Add(self.tSizeZ)

        gsizer.Add(wx.StaticText(self, -1, 'Size T:'))
        self.tSizeT = wx.TextCtrl(self, -1, str(st))
        gsizer.Add(self.tSizeT)

        gsizer.Add(wx.StaticText(self, -1, 'Size C:'))
        self.tSizeC = wx.TextCtrl(self, -1, str(sc))
        gsizer.Add(self.tSizeC)
        
        vsizer.Add(gsizer)
        
        self.stError = wx.StaticText(self, -1, '')
        vsizer.Add(self.stError)
        
        bsizer = self.CreateButtonSizer(wx.OK|wx.CANCEL)
        
        vsizer.Add(bsizer)
        
        self.SetSizerAndFit(vsizer)
        
def redimension(parent, img):
    with RedimensionDialog(parent, img) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            from PYME.IO.image import ImageStack
            from PYME.DSView import ViewIm3D
            from PYME.IO.DataSources.BaseDataSource import XYZTCWrapper

            
            
            d = XYZTCWrapper(img.data_xyztc)
            d.set_dim_order_and_size(dlg.cOrder.GetStringSelection(), size_z=int(dlg.tSizeZ.GetValue()),
                                     size_t=int(dlg.tSizeT.GetValue()), size_c=int(dlg.tSizeC.GetValue()))
            im = ImageStack(data=d, titleStub='Redimensioned')
            
            im.mdh.copyEntriesFrom(img.mdh)
            im.mdh['Parent'] = img.filename
            #im.mdh['Processing.CropROI'] = roi

            # if self.dsviewer.mode == 'visGUI':
            #     mode = 'visGUI'
            # else:
            #     mode = 'lite'

            dv = ViewIm3D(im, mode=parent.mode, glCanvas=parent.glCanvas, parent=wx.GetTopLevelParent(parent))

            #set scaling to (0,1)
            for i in range(im.data.shape[3]):
                dv.do.Gains[i] = 1.0
            
            
            
def Plug(dsviewer):
    dsviewer.AddMenuItem('Processing', 'Redimension', lambda e : redimension(dsviewer, dsviewer.image))