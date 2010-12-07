#!/usr/bin/python
##################
# photophysics.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

class DecayAnalyser:
    def __init__(self, visFr):
        self.colourFilter = visFr.colourFilter
        self.mdh = visFr.mdh

        ID_CALC_DECAYS = wx.NewId()
        self.visFr.extras_menu.Append(ID_CALC_DECAYS, "Estimate decay lifetimes")
        self.visFr.Bind(wx.EVT_MENU, self.OnCalcDecays, id=ID_CALC_DECAYS)

    def OnCalcDecays(self, event):
        from PYME.Analysis.BleachProfile import kinModels

        kinModels.fitDecay(self.colourFilter, self.mdh)
        kinModels.fitFluorBrightness(self.colourFilter, self.mdh)
        kinModels.fitOnTimes(self.colourFilter, self.mdh)
        


def Plug(visFr):
    '''Plugs this module into the gui'''
    DecayAnalyser(visFr)



