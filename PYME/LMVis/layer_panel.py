import wx
import wx.lib.newevent

import PYME.ui.autoFoldPanel as afp
import numpy as np

from PYME.ui import histLimits

import PYME.config

#DisplayInvalidEvent, EVT_DISPLAY_CHANGE = wx.lib.newevent.NewCommandEvent()

def CreateLayerPane(panel, visFr):
    pane = LayerPane(panel, visFr)
    panel.AddPane(pane)
    return pane

class LayerPane(afp.foldingPane):
    def __init__(self, panel, visFr):
        afp.foldingPane.__init__(self, panel, -1, caption="Layers", pinned=True)
        self.visFr = visFr
        
        self.il = wx.ImageList(16,16)
        self.il.Add(wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_TOOLBAR, (16,16)))
        
        print('Image list size: %d' % self.il.GetImageCount())
        
        pan = wx.Panel(self, -1)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.nb = wx.Notebook(pan, size=(150, 400))
        self.nb.AssignImageList(self.il)
        
        self.update()
            
        vsizer.Add(self.nb, 1, wx.ALL|wx.EXPAND, 0)
        
        #bAddLayer = wx.Button(pan, -1, 'New', style=wx.BU_EXACTFIT)
        #bAddLayer.Bind(wx.EVT_BUTTON, lambda e : self.visFr.add_layer())
        
        #vsizer.Add(bAddLayer, 0, wx.ALIGN_CENTRE, 0)
        
        pan.SetSizerAndFit(vsizer)
        self.AddNewElement(pan)
        
        self.visFr.layer_added.connect(self.update)
        
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_page_changed)
        
    def update(self, *args, **kwargs):
        self.nb.DeleteAllPages()
        for i, layer in enumerate(self.visFr.layers):
            page = layer.edit_traits(parent=self.nb, kind='subpanel')
            self.nb.AddPage(page.control, 'Layer %d' % i)
            
        self.nb.AddPage(wx.Panel(self.nb), 'New', imageId=0)
        
    def on_page_changed(self, event):
        if event.GetSelection() == (self.nb.GetPageCount() -1):
            #We have selected the 'new' page
            
            wx.CallAfter(self.visFr.add_layer)
        else:
            event.Skip()
        
        