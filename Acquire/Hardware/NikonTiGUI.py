#!/usr/bin/python

##################
# FilterWheel.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

class TiPanel(wx.Panel):    
    def __init__(self, parent, filterchanger, lightpath):
        self.filterchanger = filterchanger
        self.lightpath = lightpath
        
        wx.Panel.__init__(self, id=-1, parent=parent)
        #self.SetClientSize(wx.Size(148, 38))

        #vsizer = wx.BoxSizer(wx.VERTICAL)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Dichroic:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL,2)

        self.chFilters = wx.Choice(choices=self.filterchanger.names, id=-1, parent=self)
        self.chFilters.Bind(wx.EVT_CHOICE, self.OnFilterChoice)
        self.chFilters.SetSelection(self.filterchanger.GetPosition())
        hsizer.Add(self.chFilters, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        #vsizer.Add(hsizer, 0, 0,0)
        
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)        
        hsizer.Add(wx.StaticText(self, -1, 'Port:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL,2)

        self.chLightpath = wx.Choice(choices=self.lightpath.names, id=-1, parent=self)
        self.chLightpath.Bind(wx.EVT_CHOICE, self.OnPortChoice)
        self.chLightpath.SetSelection(self.lightpath.GetPosition())
        hsizer.Add(self.chLightpath, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        #vsizer.Add(hsizer, 0, 0,0)
        
        self.SetSizerAndFit(hsizer)
        
            
    def OnFilterChoice(self, event):
        n = self.chFilters.GetSelection()
        self.filterchanger.SetPosition(n)
        
    def OnPortChoice(self, event):
        n = self.chLightpath.GetSelection()
        self.lightpath.SetPosition(n)
        
    def SetSelections(self, event=None):
        self.chFilters.SetSelection(self.filterchanger.GetPosition())
        self.chLightpath.SetSelection(self.lightpath.GetPosition())
        
   
        
