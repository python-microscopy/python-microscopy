#!/usr/bin/python

##################
# displaySettingsPanel.py
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

import wx
from PYME.ui import histLimits

class dispSettingsFrame(wx.Frame):
    def __init__(self, parent, vp):
        wx.Frame.__init__(self, parent, title='Display')

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.dispPanel = dispSettingsPanel(self, vp)

        hsizer.Add(self.dispPanel, 0, wx.EXPAND|wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)


class dispSettingsPanel(wx.Panel):
    def __init__(self, parent, vp):
        wx.Panel.__init__(self, parent)

        #vsizer = wx.BoxSizer(wx.VERTICAL)

        self.vp = vp

        #self.ds = vp.ds
        if 'ds' in dir(vp.do):
            self.dsa = self.vp.do.ds
            self.vp.do.Optimise()
        else:
            self.dsa = self.vp.dsa#[:,:,0].ravel()
            self.vp.do.Optimise(self.vp.ds)


        print((self.dsa.size))
        #self.do = vp.do
        

        self.scale = 2

        #self.vp.do.Optimise(self.vp.ds)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.hlDispMapping = histLimits.HistLimitPanel(self, -1, self.dsa, self.vp.do.getDisp1Off(), self.vp.do.getDisp1Off() + 255./self.vp.do.getDisp1Gain(), True, size=(100, 80))
        self.hlDispMapping.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnMappingChange)

        hsizer.Add(self.hlDispMapping, 1, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        vsizer2 = wx.BoxSizer(wx.VERTICAL)
        self.bOptimise = wx.Button(self, -1, 'Optimise', style=wx.BU_EXACTFIT)
        self.bOptimise.Bind(wx.EVT_BUTTON, self.OnBOptimise)
        vsizer2.Add(self.bOptimise, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)


        self.cbAutoOptimise = wx.CheckBox(self, -1, 'Auto')
        self.cbAutoOptimise.SetValue(True)
        #self.cbAutoOptimise.Bind(wx.EVT_CHECKBOX, self.OnCbAutoOpt)
        vsizer2.Add(self.cbAutoOptimise, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 3)
        vsizer2.Add((0,0), 1, 0, 0)

        #hsizer.Add(vsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        #vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)
        
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)

        #scaleSizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, "Scale"), wx.HORIZONTAL)
        self.cbScale = wx.ComboBox(self, -1, choices=["1:4", "1:2", "1:1", "2:1", "4:1"], style=wx.CB_DROPDOWN|wx.CB_READONLY, size=(55, -1))
        self.cbScale.Bind(wx.EVT_COMBOBOX, self.OnScaleChanged)
        self.cbScale.SetSelection(self.scale)

        #scaleSizer.Add(self.cbScale, 0, wx.ALL, 5)
        #hsizer.Add(wx.StaticText(self, -1, 'Scale: '), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        vsizer2.Add(self.cbScale, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 5)

        hsizer.Add(vsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        #vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)

    def OnMappingChange(self, event):
        print('mc')
        lower, upper = self.hlDispMapping.GetValue()

        off = 1.*(lower)
        gain = 255./(upper - lower)

        self.vp.do.setDisp3Gain(gain)
        self.vp.do.setDisp2Gain(gain)
        self.vp.do.setDisp1Gain(gain)

        self.vp.do.setDisp3Off(off)
        self.vp.do.setDisp2Off(off)
        self.vp.do.setDisp1Off(off)

        self.vp.Refresh()
        self.vp.Update()

    def OnScaleChanged(self, event):
        self.scale = self.cbScale.GetSelection()

        self.vp.SetScale(self.scale)

    def OnBOptimise(self, event):
        self.vp.do.Optimise(self.vp.ds)

        self.hlDispMapping.SetValue((self.vp.do.getDisp1Off(), self.vp.do.getDisp1Off() + 255./self.vp.do.getDisp1Gain()))
        self.vp.Refresh()

    #def OnCBAutoOpt(self, event):

    def RefrData(self, caller=None):
        #if self.hlDispMapping.dragging == None:
        self.dsa = self.vp.dsa

        #only perform histogramming on a subset of data points to improve performance
        ##note that this may result in strange behaviour of auto-optimise
        if self.dsa.size > 1000:
            self.dsa = self.dsa[::(int(self.dsa.size/1000))]

        self.hlDispMapping.SetData(self.dsa, self.hlDispMapping.limit_lower, self.hlDispMapping.limit_upper)

        if self.cbAutoOptimise.GetValue():
            #self.OnBOptimise(None)
            self.hlDispMapping.SetValueAndFire((self.dsa.min(), self.dsa.max()))

    def __getattr__(self, name):
        if name in dir(self.hlDispMapping):
            return self.hlDispMapping.__dict__[name]
        else:  raise AttributeError(name)  # <<< DON'T FORGET THIS LINE !!

class dispSettingsPanel2(wx.Panel):
    def __init__(self, parent, vp):
        wx.Panel.__init__(self, parent)


        self.do = vp.do
        
        self.dsa = self.do.ds
        self.do.Optimise()
        


        self.scale = 2

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.hlDispMapping = histLimits.HistLimitPanel(self, -1, self.dsa, self.do.Offs[0], self.do.Offs[0] + 255./self.do.Gains[0], True, size=(100, 80))
        self.hlDispMapping.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnMappingChange)

        hsizer.Add(self.hlDispMapping, 1, wx.EXPAND|wx.ALL, 5)

        vsizer2 = wx.BoxSizer(wx.VERTICAL)
        self.bOptimise = wx.Button(self, -1, 'Optimise', style=wx.BU_EXACTFIT)
        self.bOptimise.Bind(wx.EVT_BUTTON, self.OnBOptimise)
        vsizer2.Add(self.bOptimise, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)


        self.cbAutoOptimise = wx.CheckBox(self, -1, 'Auto')
        self.cbAutoOptimise.SetValue(True)
        #self.cbAutoOptimise.Bind(wx.EVT_CHECKBOX, self.OnCbAutoOpt)
        vsizer2.Add(self.cbAutoOptimise, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 3)
        vsizer2.Add((0,0), 1, 0, 0)

        #hsizer.Add(vsizer2, 0, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        #vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        #hsizer = wx.BoxSizer(wx.HORIZONTAL)

        #scaleSizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, "Scale"), wx.HORIZONTAL)
        self.cbScale = wx.ComboBox(self, -1, choices=["1:4", "1:2", "1:1", "2:1", "4:1"], style=wx.CB_DROPDOWN|wx.CB_READONLY, size=(55, -1))
        self.cbScale.Bind(wx.EVT_COMBOBOX, self.OnScaleChanged)
        self.cbScale.SetSelection(self.scale)

        #scaleSizer.Add(self.cbScale, 0, wx.ALL, 5)
        #hsizer.Add(wx.StaticText(self, -1, 'Scale: '), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        vsizer2.Add(self.cbScale, 0, wx.EXPAND, 5)

        hsizer.Add(vsizer2, 0, wx.EXPAND|wx.ALL, 5)

        #vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        self.SetSizer(hsizer)
        hsizer.Fit(self)

    def OnMappingChange(self, event):
        lower, upper = self.hlDispMapping.GetValue()

        self.do.SetOffset(0, lower)
        self.do.SetGain(0,1./(upper- lower))

        #self.vp.Refresh()

    def OnScaleChanged(self, event):
        self.scale = self.cbScale.GetSelection()

        self.do.SetScale(self.scale - 2)

    def OnBOptimise(self, event):
        self.do.Optimise()

        self.hlDispMapping.SetValue((self.do.Offs[0], self.do.Offs[0] + 1./self.do.Gains[0]))
        #self.vp.Refresh()

    #def OnCBAutoOpt(self, event):

    def RefrData(self, caller=None):
        #if self.hlDispMapping.dragging == None:
        self.dsa = self.do.ds[:,:,0].ravel('F')

        #only perform histogramming on a subset of data points to improve performance
        ##note that this may result in strange behaviour of auto-optimise
        if self.dsa.size > 1000:
            self.dsa = self.dsa[::(int(self.dsa.size/1000))]

        self.hlDispMapping.SetData(self.dsa, self.hlDispMapping.limit_lower, self.hlDispMapping.limit_upper)

        if self.cbAutoOptimise.GetValue():
            #self.OnBOptimise(None)
            self.hlDispMapping.SetValueAndFire((self.dsa.min(), self.dsa.max()))

    def __getattr__(self, name):
        if name in dir(self.hlDispMapping):
            return self.hlDispMapping.__dict__[name]
        else:  raise AttributeError(name)  # <<< DON'T FORGET THIS LINE !!
