#!/usr/bin/python
##################
# pointwiseColoc.py
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#import wx

class PointwiseColocaliser:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Analysis', "Pointwise Colocalisation", self.OnPointwiseColoc)
        
    def OnPointwiseColoc(self, event):
        import wx
        #from PYME import mProfile
        #mProfile.profileOn(['distColoc.py'])
        from PYME.Analysis.Colocalisation import distColoc
        from PYME.DSView.modules.coloc import ColocSettingsDialog

        chans = self.visFr.pipeline.colourFilter.getColourChans()

        if len(chans) == 0:
            md = wx.MessageDialog(self.visFr, 'Not enough colour channels', 
                                  'Pointwise colocalization requires 2 colour channels', wx.OK)
            md.ShowModal()
            return

        dlg = ColocSettingsDialog(self.visFr, names=chans, show_bins=False)
        
        if dlg.ShowModal() == wx.ID_OK:

            selected_chans = [chans[int(x)] for x in dlg.GetChans()]  # re-index into chans b/c calcDistCorr expects full channel names

            #A vs B
            distColoc.calcDistCorr(self.visFr.pipeline.colourFilter, *selected_chans)
            #B vs A
            #distColoc.calcDistCorr(self.visFr.colourFilter, *(self.visFr.colourFilter.getColourChans()[::-1]))
            #mProfile.report()

        dlg.Destroy()


def Plug(visFr):
    '''Plugs this module into the gui'''
    PointwiseColocaliser(visFr)


