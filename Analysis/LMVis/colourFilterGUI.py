#!/usr/bin/python
##################
# colourFilterGUI.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import wx
import PYME.misc.autoFoldPanel as afp

def CreateColourFilterPane(panel, mapping, visFr):
    pane = ColourFilterPane(panel, mapping, visFr)
    panel.AddPane(pane)
    return pane

class ColourFilterPane(afp.foldingPane):
    def __init__(self, panel, colourFilter, visFr):
        afp.foldingPane.__init__(self, panel, -1, caption="Colour", pinned = True)

        self.colourFilter = colourFilter
        self.visFr = visFr

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


    def UpdateColourFilterChoices(self):
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
        self.visFr.ClearGenerated()


