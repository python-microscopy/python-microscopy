#!/usr/bin/python
##################
# profilePlotting.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx

#from PYME.Acquire.mytimer import mytimer
import pylab
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer_npy_nb import ViewIm3D, ImageStack

class fitter:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        #self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image

        fit_menu = wx.Menu()

        FIT_RAW_INTENSITY_DECAY = wx.NewId()
        
        fit_menu.Append(FIT_RAW_INTENSITY_DECAY, "Raw Intensity Decay", "", wx.ITEM_NORMAL)

        dsviewer.menubar.Insert(dsviewer.menubar.GetMenuCount()-1, fit_menu, 'Fitting')

        wx.EVT_MENU(dsviewer, FIT_RAW_INTENSITY_DECAY, self.OnRawDecay)
        
    def OnRawDecay(self, event):
        from PYME.Analysis.BleachProfile import rawIntensity
        I = self.image.data[:,0,0].squeeze()
        imo = self.image.parent
        
        rawIntensity.processIntensityTrace(I, imo.mdh, dt=imo.mdh['Camera.CycleTime'])
        




def Plug(dsviewer):
    dsviewer.fitter = fitter(dsviewer)
    
