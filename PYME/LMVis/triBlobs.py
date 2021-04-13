# -*- coding: utf-8 -*-
"""
Created on Sat May 14 12:40:45 2016

@author: david
"""
import wx
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import numpy as np

try:
    from PYME.ui import recArrayView
except:
    pass

try:
    from enthought.traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, on_trait_change
    #from enthought.traits.ui.api import View, Item, EnumEditor, InstanceEditor, Group
except ImportError:
    from traits.api import HasTraits, Float, File, BaseEnum, Enum, List, Instance, CStr, Bool, Int, on_trait_change
    #from traitsui.api import View, Item, EnumEditor, InstanceEditor, Group

class BlobSettings(HasTraits):
    distThreshold = Float(30.)
    minSize = Int(10.)
    jittering = Float(0)
    blobColourKey = CStr('Index')

def GenBlobPanel(visgui, pnl):
#        item = self._pnl.AddFoldPanel("Objects", collapsed=False,
#                                      foldIcons=self.Images)
        item = afp.foldingPane(pnl, -1, caption="Objects", pinned = True)

        pan = wx.BlobSettingsPanel(item, visgui.pipeline, visgui.pipeline.blobSettings, visgui)
        
         #self._pnl.AddFoldPanelWindow(item, pan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 5)
        item.AddNewElement(pan)

        pnl.AddPane(item)
        
class BlobSettingsPanel(wx.Panel):
    def __init__(self, pipeline, blobSettings, visgui):
        wx.Panel.__init__(self, -1)

        self.pipeline = pipeline
        self.blobSettings = blobSettings
        self.visgui = visgui        
        
        bsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Threshold [nm]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tBlobDist = wx.TextCtrl(self, -1, '%3.0f' % self.blobSettings.distThreshold,size=(40,-1))
        hsizer.Add(self.tBlobDist, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Min Size [events]:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tMinObjSize = wx.TextCtrl(self, -1, '%d' % self.blobSettings.minSize, size=(40, -1))
        hsizer.Add(self.tMinObjSize, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Jittering:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tObjJitter = wx.TextCtrl(self, -1, '%d' % self.blobSettings.jittering, size=(40, -1))
        hsizer.Add(self.tObjJitter, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        self.bApplyThreshold = wx.Button(self, -1, 'Apply')
        bsizer.Add(self.bApplyThreshold, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.bObjMeasure = wx.Button(self, -1, 'Measure')
        #self.bObjMeasure.Enable(False)
        bsizer.Add(self.bObjMeasure, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Object Colour:'), 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.cBlobColour = wx.Choice(self, -1, choices=['Index', 'Random'])
        self.cBlobColour.SetSelection(0)
        self.cBlobColour.Bind(wx.EVT_CHOICE, self.OnSetBlobColour)

        hsizer.Add(self.cBlobColour, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        bsizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

        self.SetSizerAndFit(bsizer)
        
        self.bApplyThreshold.Bind(wx.EVT_BUTTON, self.OnObjApplyThreshold)
        self.bObjMeasure.Bind(wx.EVT_BUTTON, self.OnObjMeasure)



       

    def OnSetBlobColour(self, event):
        bcolour = self.cBlobColour.GetStringSelection()
        
        self.blobSettings.blobColourKey = bcolour

        if bcolour == 'Index':
            c = self.objCInd.astype('f')
        elif bcolour == 'Random':
            r = np.random.rand(self.objCInd.max() + 1)
            c = r[self.objCInd.astype('i')]
        else:
            c = self.pipeline.objectMeasures[bcolour][self.objCInd.astype('i')]

        self.visgui.glCanvas.c = c
        self.visgui.glCanvas.setColour()
        self.visgui.OnGLViewChanged()
        
        self.visgui.displayPane.hlCLim.SetData(self.glCanvas.c, self.glCanvas.clim[0], self.glCanvas.clim[1])

    def OnObjApplyThreshold(self, event):
        self.pipeline.objects = None
        
        self.blobSettings.distThreshold = float(self.tBlobDist.GetValue())
        self.blobSettings.minSize = int(self.tMinObjSize.GetValue())
        self.blobSettings.jittering = int(self.tObjJitter.GetValue())

        #self.bObjMeasure.Enable(True)

        #self.RefreshView()

    def OnObjMeasure(self, event):
        om = self.pipeline.measureObjects()
        
        if self.visgui.rav is None:
            self.visgui.rav = recArrayView.ArrayPanel(self.visgui, om)
            self.visgui.AddPage(self.visgui.rav, 'Measurements')
        else:
            self.visgui.rav.grid.SetData(om)

        self.cBlobColour.Clear()
        self.cBlobColour.Append('Index')

        for n in om.dtype.names:
            self.cBlobColour.Append(n)

        self.visgui.RefreshView()
