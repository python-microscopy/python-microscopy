#!/usr/bin/python

##################
# genImageDialog.py
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
from . import histLimits


class GenImageDialog(wx.Dialog):
    def __init__(self, parent, mode='current', defaultPixelSize=5.0, jitterVariables = [], jitterVarDefault=0, mcProbDefault = 1.0, colours = [], zvals=None, jitterVarDefaultZ=0):
        wx.Dialog.__init__(self, parent, title='Generate Image ...')

        pixelSzs = ['%3.2f' % (defaultPixelSize*2**n) for n in range(6)]

        jitterPhrase = 'Jitter [nm]:'
        if mode in ['gaussian', '3Dgaussian']:
            jitterPhrase = 'Std. Dev. [nm]:'
        
        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        
        #we always want a pixel size
        sizer2.Add(wx.StaticText(self, -1, 'Pixel Size [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        self.cbPixelSize = wx.ComboBox(self, -1, value='%3.2f' % defaultPixelSize, choices=pixelSzs, style=wx.CB_DROPDOWN, size=(150, -1))


        sizer2.Add(self.cbPixelSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        
        if not zvals == None:
            nZlevels = len(set(zvals))
        else:
            nZlevels = 0

        #jitter parameter for gaussian and triangles
        if mode in ['gaussian', 'triangles', 'trianglesw', '3Dgaussian', '3Dtriangles']:
            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            sizer2.Add(wx.StaticText(self, -1, jitterPhrase), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            self.tJitterScale = wx.TextCtrl(self, -1, '1.0', size=(60, -1))
            sizer2.Add(self.tJitterScale, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            sizer2.Add(wx.StaticText(self, -1, 'x'), 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP|wx.BOTTOM, 5)
    
            self.cJitterVariable = wx.Choice(self, -1, choices=jitterVariables, size=(150, -1))
            self.cJitterVariable.SetSelection(jitterVarDefault)
            sizer2.Add(self.cJitterVariable, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        #jitter parameter for gaussian in z
        if mode in  ['3Dgaussian', '3Dtriangles']:
            zJitSc = 1.0
            if nZlevels < 100: #stack rather than 3D fit
                zJitSc = 200.
                jitterVarDefaultZ = 0
                
            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            sizer2.Add(wx.StaticText(self, -1, 'Z Std. Dev. [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            self.tJitterScaleZ = wx.TextCtrl(self, -1, '%3.2f'% zJitSc, size=(60, -1))
            sizer2.Add(self.tJitterScaleZ, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            sizer2.Add(wx.StaticText(self, -1, 'x'), 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP|wx.BOTTOM, 5)

            self.cJitterVariableZ = wx.Choice(self, -1, choices=jitterVariables, size=(150, -1))
            self.cJitterVariableZ.SetSelection(jitterVarDefaultZ)
            sizer2.Add(self.cJitterVariableZ, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        #MC subsampling parameter for triangles
        if mode in ['triangles', 'trianglesw', '3Dtriangles']:
            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            sizer2.Add(wx.StaticText(self, -1, 'MC subsampling probability:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            self.tMCProb = wx.TextCtrl(self, -1, '%3.2f' % mcProbDefault, size=(60, -1))
            sizer2.Add(self.tMCProb, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            sizer2.Add(wx.StaticText(self, -1, '# Samples to average:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            self.tNumSamps = wx.TextCtrl(self, -1, '10', size=(60, -1))
            sizer2.Add(self.tNumSamps, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            #sizer2.Add(wx.StaticText(self, -1, 'Use software rendereing:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        if mode in ['triangles', 'trianglesw']:
            self.cbTriangSoftRender = wx.CheckBox(self, -1, 'Use software rendering')
            self.cbTriangSoftRender.SetValue(True)
            sizer2.Add(self.cbTriangSoftRender, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        if mode in ['3Dhistogram', '3Dgaussian', '3Dtriangles']:
            zThick = 50
            if nZlevels < 100: #stack rather than 3D fit
                zThick = 200
                
            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            sizer2.Add(wx.StaticText(self, -1, 'Z slice thickness [nm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            self.tZThickness = wx.TextCtrl(self, -1, '%d' % zThick, size=(60, -1))
            sizer2.Add(self.tZThickness, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

            sizer2 = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Z range:'), wx.VERTICAL)

            self.hZRange = histLimits.HistLimitPanel(self, -1, zvals[::max(len(zvals)/1e4, 1)], zvals.min() - 50, zvals.max() + 50, size=(150, 80))
            self.hZRange.binSize = float(self.tZThickness.GetValue())

            self.tZThickness.Bind(wx.EVT_TEXT, self.OnZBinChange)

            sizer2.Add(self.hZRange, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL|wx.EXPAND, 5)

            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        #multiple colour channels
        if len(colours) > 0:
            sizer2 = wx.BoxSizer(wx.HORIZONTAL)
            sizer2.Add(wx.StaticText(self, -1, 'Colour[s]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

            self.colours = ['Everything'] + colours

            self.lColour = wx.ListBox(self, -1, choices=self.colours, size=(150, -1), style=wx.LB_MULTIPLE)

            for n in range(1, len(self.colours)):
                self.lColour.SetSelection(n)

            sizer2.Add(self.lColour, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
            sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL|wx.EXPAND, 5)

        
        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnZBinChange(self, event):
        self.hZRange.binSize = float(self.tZThickness.GetValue())
        self.hZRange.Refresh()

    def getPixelSize(self):
        return float(self.cbPixelSize.GetValue())

    def getJitterVariable(self):
        return self.cJitterVariable.GetStringSelection()

    def getJitterScale(self):
        return float(self.tJitterScale.GetValue())

    def getJitterVariableZ(self):
        return self.cJitterVariableZ.GetStringSelection()

    def getJitterScaleZ(self):
        return float(self.tJitterScaleZ.GetValue())

    def getMCProbability(self):
        return float(self.tMCProb.GetValue())

    def getNumSamples(self):
        return int(self.tNumSamps.GetValue())

    def getColour(self):
        if 'colours' in dir(self):
            return [self.lColour.GetString(n).encode() for n in self.lColour.GetSelections()]
        else:
            return [None]

    def getZSliceThickness(self):
        if 'tZThickness' in dir(self):
            return float(self.tZThickness.GetValue())
        else:
            return 0

    def getZBounds(self):
        return self.hZRange.GetValue()

    def getSoftRender(self):
        return self.cbTriangSoftRender.GetValue()
