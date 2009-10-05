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

#Boa:Frame:FiltFrame

import wx
from fw102 import FW102B as filtWheel

[wxID_FILTFRAME, wxID_FILTFRAMECHFILTWHEEL, wxID_FILTFRAMEPANEL1, 
] = [wx.NewId() for _init_ctrls in range(3)]

class WFilter:
    def __init__(self, pos, name, description, OD=None):
        '''Create a filter object given position in wheel, a short name to use in
        for filter selection, a textual description,
        and the optical density of the filter (if a neutral density filter - None
        otherwise)'''
        self.pos = pos
        self.name = name
        self.description = description
        self.OD = OD

class FiltFrame(wx.Panel):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Panel.__init__(self, id=wxID_FILTFRAME,
              parent=prnt, size=wx.Size(156, 38))
        #self.SetClientSize(wx.Size(148, 38))

        self.panel1 = wx.Panel(id=wxID_FILTFRAMEPANEL1, name='panel1',
              parent=self, pos=wx.Point(0, 0), size=wx.Size(148, 38),
              style=wx.TAB_TRAVERSAL)

        self.chFiltWheel = wx.Choice(choices=[], id=wxID_FILTFRAMECHFILTWHEEL,
              name=u'chFiltWheel', parent=self.panel1, pos=wx.Point(8, 8),
              size=wx.Size(128, 21), style=0)
        self.chFiltWheel.Bind(wx.EVT_CHOICE, self.OnChFiltWheelChoice,
              id=wxID_FILTFRAMECHFILTWHEEL)

    def __init__(self, parent, installedFilters, serPort='COM3'):
        '''Create a filter wheel gui object. installedFilters should be a list of
        WFilter objects. The first item is the default'''
        self.installedFilters = installedFilters 
        self._init_ctrls(parent)
        
        self.fw = filtWheel(serPort)

        for k in installedFilters:
            self.chFiltWheel.Append(k.name)
            
        self.chFiltWheel.SetSelection(0)
        self.fw.setPos(installedFilters[0].pos)
            
    def OnChFiltWheelChoice(self, event):
        n = self.chFiltWheel.GetSelection()
        self.fw.setPos(self.installedFilters[n].pos)
        
    def SetFilterPos(self, name = None, id = None, pos = None):
        '''Set filter by either name, position in list, or postion
        in filter wheel'''
        if not name == None:
            id = [n for n, f in zip(range(len(self.installedFilters), self.installedFilters)) if f.name == name][0]
        elif id == None:
            id = [n for n, f in zip(range(len(self.installedFilters), self.installedFilters)) if f.pos == pos][0]
            
        self.fw.setPos(self.installedFilters[id].pos)
        self.chFiltWheel.SetSelection(id)
        
    def GetSelectedFilter(self):
        n = self.chFiltWheel.GetSelection()
        return self.installedFilters(n)
        
