#!/usr/bin/python
##################
# colourFilterGUI.py
#
# Copyright David Baddeley, 2010
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
#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp

def CreateColourFilterPane(panel, mapping, visFr):
    pane = ColourFilterPane(panel, mapping, visFr)
    panel.AddPane(pane)
    return pane

class ColourFilterPane(afp.foldingPane):
    def __init__(self, panel, colourFilter, pipeline):
        afp.foldingPane.__init__(self, panel, -1, caption="Colour", pinned = True)

        self.colourFilter = colourFilter
        self.pipeline = pipeline

        cnames = ['Everything']

        if self.colourFilter:
            cnames += self.colourFilter.getColourChans()

        self.chColourFilterChan = wx.Choice(self, -1, choices=cnames, size=(170, -1))

        if self.colourFilter and self.colourFilter.currentColour in cnames:
            self.chColourFilterChan.SetSelection(cnames.index(self.colourFilter.currentColour))
        else:
            self.chColourFilterChan.SetSelection(0)

        self.chColourFilterChan.Bind(wx.EVT_CHOICE, self.OnColourFilterChange)
        #self._pnl.AddFoldPanelWindow(self, self.chColourFilterChan, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
        self.AddNewElement(self.chColourFilterChan)
        
        self.pipeline.onRebuild.connect(self.UpdateColourFilterChoices)


    def UpdateColourFilterChoices(self, **kwargs):
        cnames = ['Everything']

        if self.colourFilter:
            cnames += self.colourFilter.getColourChans()

        self.chColourFilterChan.Clear()
        for cn in cnames:
            self.chColourFilterChan.Append(cn)

        if self.colourFilter and self.colourFilter.currentColour in cnames:
            self.chColourFilterChan.SetSelection(cnames.index(self.colourFilter.currentColour))
        else:
            self.chColourFilterChan.SetSelection(0)


    def OnColourFilterChange(self, event):
        self.colourFilter.setColour(event.GetString())
        self.pipeline.ClearGenerated()


