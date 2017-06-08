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
        
        pan = wx.Panel(self, -1)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.nb = wx.Notebook(pan, size=(150, 400))
        
        self.update()
            
        vsizer.Add(self.nb, 1, wx.ALL|wx.EXPAND, 0)
        
        bAddLayer = wx.Button(pan, -1, 'New', style=wx.BU_EXACTFIT)
        bAddLayer.Bind(wx.EVT_BUTTON, lambda e : self.visFr.add_layer())
        
        vsizer.Add(bAddLayer, 0, wx.ALIGN_CENTRE, 0)
        
        pan.SetSizerAndFit(vsizer)
        self.AddNewElement(pan)
        
        self.visFr.layer_added.connect(self.update)
        
    def update(self, *args, **kwargs):
        self.nb.DeleteAllPages()
        for i, layer in enumerate(self.visFr.layers):
            page = layer.edit_traits(parent=self.nb, kind='subpanel')
            self.nb.AddPage(page.control, 'Layer %d' % i)
        
        
        