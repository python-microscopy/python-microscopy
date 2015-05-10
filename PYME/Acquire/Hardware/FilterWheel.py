#!/usr/bin/python

##################
# FilterWheel.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#Boa:Frame:FiltFrame

import wx
from .fw102 import FW102B as filtWheel

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
        
class FiltWheel(object):
    def __init__(self, installedFilters, serPort='COM3', dichroic=None):
        '''Create a filter wheel gui object. installedFilters should be a list of
        WFilter objects. The first item is the default'''
        self.installedFilters = installedFilters 
        
        self.fw = filtWheel(serPort)
        if dichroic:
            self.dichroic = dichroic
            self.dichroic.wantChangeNotification.append(self.DichroicSync)
            self.DICHROIC_SYNC = True

        #self.fw.setPos(installedFilters[0].pos)
        
    def SetFilterPos(self, name = None, id = None, pos = None):
        '''Set filter by either name, position in list, or postion
        in filter wheel'''
        if not name == None:
            id = [n for n, f in enumerate(self.installedFilters) if f.name == name][0]
        elif id == None:
            self.fw.setPos(pos)
            return
            #id = [n for n, f in enumerate(self.installedFilters) if f.pos == pos][0]
            
        self.fw.setPos(self.installedFilters[id].pos)
        
        
    def GetFilterNames(self):
        return [f.name for f in self.installedFilters]
        
    def GetCurrentIndex(self):
        p = self.fw.getPos()
        for i, f in enumerate(self.installedFilters):
            if f.pos == p:
                return i
                
    def DichroicSync(self):
        if self.DICHROIC_SYNC:
            dname =  self.dichroic.GetFilter()
            
            if dname in self.GetFilterNames():
                self.SetFilterPos(name=dname)
        

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

    def __init__(self, parent, filterWheel):
        '''Create a filter wheel gui object. installedFilters should be a list of
        WFilter objects. The first item is the default'''
        self._init_ctrls(parent)
        
        self.fWheel = filterWheel

        for k in self.fWheel.installedFilters:
            self.chFiltWheel.Append(k.name)
            
        self.chFiltWheel.SetSelection(self.fWheel.GetCurrentIndex())
        #self.fw.setPos(installedFilters[0].pos)
            
    def OnChFiltWheelChoice(self, event):
        n = self.chFiltWheel.GetSelection()
        self.fWheel.SetFilterPos(id=n)
        
    def GetSelectedFilter(self):
        n = self.chFiltWheel.GetSelection()
        return self.fWheel.installedFilters(n)
        
    
        
