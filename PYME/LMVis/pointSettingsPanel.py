# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:20:11 2016

@author: david
"""
import wx
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp

from PYME.recipes.traits import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, on_trait_change

class PointDisplaySettings(HasTraits):
    pointSize = Float(5.0)
    colourDataKey = CStr('t')
    alpha = Float(1.0)
    
    
def _getPossibleKeys(pipeline):
    colKeys = ['<None>']

    if not pipeline.colourFilter is None: #is the test needed?
        colKeys += list(pipeline.keys())

    colKeys += list(pipeline.GeneratedMeasures.keys())

    colKeys.sort()
    
    return colKeys


class PointSettingsPanel(wx.Panel):
    """A GUI class for determining the settings to use when displaying points
    in VisGUI.
    
    Constructed as follows: 
    PointSettingsPanel(parent, pipeline, pointDisplaySettings)
    
    where: 
      parent is the parent window
      pipeline is the pipeline object which provides the points,
      pointDisplaySettings is an instance of PointDisplaySettings
    
    
    """
    
    def __init__(self, parent, pipeline, pointDisplaySettings):
        wx.Panel.__init__(self, parent, -1)
        
        self.pipeline = pipeline
        self.pointDisplaySettings = pointDisplaySettings
        
        
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Size [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPointSize = wx.TextCtrl(self, -1, '%3.2f' % self.pointDisplaySettings.pointSize)
        hsizer.Add(self.tPointSize, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Alpha:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.tPointAlpha = wx.TextCtrl(self, -1, '%3.2f' % self.pointDisplaySettings.alpha)
        hsizer.Add(self.tPointAlpha, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        
        colKeys = _getPossibleKeys(self.pipeline)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Colour:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.chPointColour = wx.Choice(self, -1, choices=colKeys)
        
        currentCol = self.pointDisplaySettings.colourDataKey
        if currentCol in colKeys:
            self.chPointColour.SetSelection(colKeys.index(currentCol))
        
        
        hsizer.Add(self.chPointColour, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)
        
        self.SetSizerAndFit(bsizer)


        self.tPointSize.Bind(wx.EVT_TEXT, self.OnPointSizeChange)
        self.tPointAlpha.Bind(wx.EVT_TEXT, self.OnPointAlphaChange)
        self.chPointColour.Bind(wx.EVT_CHOICE, self.OnChangePointColour)
        self.chPointColour.Bind(wx.EVT_ENTER_WINDOW, self.UpdatePointColourChoices)

        self.pipeline.onKeysChanged.connect(self.UpdatePointColourChoices)

        

    def UpdatePointColourChoices(self, event=None, **kwargs):
        """Update our choice of keys if the pipeline has changed.
        """
        colKeys = _getPossibleKeys(self.pipeline)

        self.chPointColour.Clear()
        self.chPointColour.SetItems(colKeys)

        currentCol = self.pointDisplaySettings.colourDataKey
        if currentCol in colKeys:
            self.chPointColour.SetSelection(colKeys.index(currentCol))

    def OnPointSizeChange(self, event):
        self.pointDisplaySettings.pointSize = float(self.tPointSize.GetValue())
        #self.glCanvas.Refresh()

    def OnPointAlphaChange(self, event):
        try:
            self.pointDisplaySettings.alpha = float(self.tPointAlpha.GetValue())
        except ValueError:
            pass  # do nothing, if value couldn't be parsed
        #self.glCanvas.Refresh()

    def OnChangePointColour(self, event):
        self.pointDisplaySettings.colourDataKey = event.GetString()
        
        
        
def GenPointsPanel(visgui, pnl, title='Points'):
    """Generate a ponts pane and insert into the given panel"""
    item = afp.foldingPane(pnl, -1, caption=title, pinned = True)

    pan = PointSettingsPanel(item, visgui.pipeline, visgui.pointDisplaySettings)
    item.AddNewElement(pan)
    pnl.AddPane(item)
