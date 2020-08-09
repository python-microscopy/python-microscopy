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
        #from PYME import mProfile
        #mProfile.profileOn(['distColoc.py'])
        from PYME.Analysis.Colocalisation import distColoc
        #A vs B
        distColoc.calcDistCorr(self.visFr.pipeline.colourFilter, *(self.visFr.pipeline.colourFilter.getColourChans()[::1]))
        #B vs A
        #distColoc.calcDistCorr(self.visFr.colourFilter, *(self.visFr.colourFilter.getColourChans()[::-1]))
        #mProfile.report()



def Plug(visFr):
    '''Plugs this module into the gui'''
    PointwiseColocaliser(visFr)


