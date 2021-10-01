# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:20:11 2016

@author: david
"""
import wx
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import numpy as np

from PYME.recipes.traits import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, on_trait_change

#from PYME.Analysis.points.QuadTree import pointQT

class QuadTreeSettings(HasTraits):
    leafSize = Int(10)
    


class QuadTreeSettingsPanel(wx.Panel):
    """A GUI class for determining the settings to use when creating a QuadTree
    in VisGUI.
    
    Constructed as follows: 
    QuadTreeSettingsPanel(parent, pipeline, quadTreeSettings)
    
    where: 
      parent is the parent window
      pipeline is the pipeline object
      quadTreeSettings is an instance of quadTreeSettings
    
    
    """
    
    def __init__(self, parent, pipeline, quadTreeSettings):
        wx.Panel.__init__(self, parent, -1)
        
        self.quadTreeSettings = quadTreeSettings
        self.pipeline = pipeline
        
        
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Leaf Size:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tQTLeafSize = wx.TextCtrl(self, -1, '%d' % self.quadTreeSettings.leafSize)
        hsizer.Add(self.tQTLeafSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        self.stQTSNR = wx.StaticText(self, -1, 'Effective SNR = %3.2f' % np.sqrt(self.quadTreeSettings.leafSize/2.0))
        bsizer.Add(self.stQTSNR, 0, wx.ALL, 5)

        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #hsizer.Add(wx.StaticText(pan, -1, 'Goal pixel size [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #self.tQTSize = wx.TextCtrl(pan, -1, '20000')
        #hsizer.Add(self.tQTLeafSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        #bsizer.Add(hsizer, 0, wx.ALL, 0)
        
        self.SetSizer(bsizer)
        bsizer.Fit(self)
        
        self.tQTLeafSize.Bind(wx.EVT_TEXT, self.OnQTLeafChange)
        
    def OnQTLeafChange(self, event):
        #from PYME.Analysis.points.QuadTree import pointQT
        
        leafSize = int(self.tQTLeafSize.GetValue())
        if not leafSize >= 1:
            raise RuntimeError('QuadTree leaves must be able to contain at least 1 item')

        self.quadTreeSettings.leafSize = leafSize
        self.stQTSNR.SetLabel('Effective SNR = %3.2f' % np.sqrt(self.quadTreeSettings.leafSize/2.0))

        #FIXME - shouldn't need to do this here
        self.pipeline.Quads = None
        #self.RefreshView()
        
        
def GenQuadTreePanel(visgui, pnl, title='Points'):
    """Generate a ponts pane and insert into the given panel"""
    item = afp.foldingPane(pnl, -1, caption=title, pinned = True)

    pan = QuadTreeSettingsPanel(item, visgui.pipeline, visgui.quadTreeSettings)
    item.AddNewElement(pan)
    pnl.AddPane(item)
