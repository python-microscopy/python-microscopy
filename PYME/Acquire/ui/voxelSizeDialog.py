#!/usr/bin/python
##################
# voxelSizeDialog.py
#
# Copyright David Baddeley, 2011
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

import wx

class VoxelSizeDialog(wx.Dialog):
    def __init__(self, parent, scope):
        wx.Dialog.__init__(self, parent, -1, 'Pixel Size Calibration')
        self.scope=scope

        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.camNames = list(scope.cameras.keys())
        self.vsChoices = []
        #self.vsChoiceIDs = []
        

        gsizer = wx.FlexGridSizer(2, 2, 2)
        gsizer.AddGrowableCol(1,1)
        for camName in self.camNames:
            gsizer.Add(wx.StaticText(self, -1, '%s: ' % camName), 0, wx.ALIGN_CENTER_VERTICAL, 0)
            ch = wx.Choice(self, -1)#, size=(150, -1))
            ch.Bind(wx.EVT_CHOICE, self.OnChooseVS)
            gsizer.Add(ch, 1, wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 0)
            self.vsChoices.append(ch)
            #self.vsChoiceIDs.append(ch.GetId())

        self._setVoxelSizeChoices()

        sizer1.Add(gsizer, 0, wx.EXPAND|wx.ALL, 5)

        hsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'New Setting'), wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Name: '), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self.tName = wx.TextCtrl(self, -1, size=(80, -1))
        hsizer.Add(self.tName, 1, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 10)

        hsizer.Add(wx.StaticText(self, -1, u'(x '), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self.tx = wx.TextCtrl(self, -1, size=(30, -1))
        hsizer.Add(self.tx, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)

        hsizer.Add(wx.StaticText(self, -1, u', y '), 0, wx.ALIGN_CENTER_VERTICAL, 0)
        self.ty = wx.TextCtrl(self, -1, size=(30, -1))
        hsizer.Add(self.ty, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)
        hsizer.Add(wx.StaticText(self, -1, u') [\u03BCm] '), 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 10)

        self.bAdd = wx.Button(self, -1, 'Add', style=wx.BU_EXACTFIT)
        self.bAdd.Bind(wx.EVT_BUTTON, self.OnAddVSSetting)
        hsizer.Add(self.bAdd, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        sizer1.Add(hsizer, 0, wx.EXPAND|wx.ALL, 5)


        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        #btn = wx.Button(self, wx.ID_CANCEL)

        #btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnAddVSSetting(self, event):
        self.scope.AddVoxelSizeSetting(self.tName.GetValue(), float(self.tx.GetValue()), float(self.ty.GetValue()))
        self._setVoxelSizeChoices()

    def OnChooseVS(self, event):
        #id = event.GetId()
        ch = event.GetEventObject()
        i = self.vsChoices.index(ch)
        camName = self.camNames[i]

        vsname = self.voxNames[ch.GetStringSelection()]

        self.scope.SetVoxelSize(vsname, camName)





    def _setVoxelSizeChoices(self):
        with self.scope.settingsDB as conn:
            voxelsizes = conn.execute("SELECT ID, name, x,y FROM VoxelSizes ORDER BY ID DESC").fetchall()
    
            voxIDs = []
            self.voxNames = {}
            #self.compNames = []
    
            for ch in self.vsChoices:
                ch.Clear()
    
            for ID, name, x, y in voxelsizes:
                compName = '%s - (%3.3f, %3.3f)' % (name, x, y)
    
                voxIDs.append(ID)
                self.voxNames[compName] = name
    
                for ch in self.vsChoices:
                    ch.Append(compName)
    
            #print voxelsizes
            #print voxIDs
    
            for ch, camName in zip(self.vsChoices, self.camNames):
                try:
                    currVoxelSizeID = conn.execute("SELECT sizeID FROM VoxelSizeHistory2 WHERE camSerial=? ORDER BY time DESC", (self.scope.cameras[camName].GetSerialNumber(),)).fetchone()[0]
        
                    #print currVoxelSizeID
                    if not currVoxelSizeID is None:
                        ch.SetSelection(voxIDs.index(currVoxelSizeID))
                except:
                    pass









