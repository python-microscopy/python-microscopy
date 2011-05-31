#!/usr/bin/python
##################
# prebleachExtraction.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
import numpy as np
#from PYME.Analysis import piecewiseMapping

class PrebleachExtractor:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image
        self.split = 'Splitter' in self.image.mdh.getEntry('Analysis.FitModule')
        self.mixmatrix=np.array([[.85, .15],[.11, .89]])

        EXTRAS_PREBLEACH = wx.NewId()
        dsviewer.mExtras.Append(EXTRAS_PREBLEACH, "&Extract Prebleach Image", "", wx.ITEM_NORMAL)
        wx.EVT_MENU(dsviewer, EXTRAS_PREBLEACH, self.OnExtract)

    def OnExtract(self, event):
        from PYME.DSView import View3D
        #print 'extracting ...'

        mdh = self.image.mdh

        #dark = deTile.genDark(self.vp.do.ds, self.image.mdh)
        dark = mdh.getEntry('Camera.ADOffset')

        #split = False

        frames = mdh.getEntry('Protocol.PrebleachFrames')
        
        dt = self.image.data[:,:,frames[0]:frames[1]].astype('f').mean(2)- dark

        ROIX1 = mdh.getEntry('Camera.ROIPosX')
        ROIY1 = mdh.getEntry('Camera.ROIPosY')

        ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
        ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

        if self.split:
            from PYME.Acquire.Hardware import splitter
            unmux = splitter.Unmixer([mdh.getEntry('chroma.dx'),mdh.getEntry('chroma.dy')], 1e3*mdh.getEntry('voxelsize.x'))

            dt = unmux.Unmix(dt, self.mixmatrix, 0, [ROIX1, ROIY1, ROIX2, ROIY2])

            View3D(dt, 'Prebleach Image')
        else:
            View3D(dt, 'Prebleach Image')


def Plug(dsviewer):
    dsviewer.pbe = PrebleachExtractor(dsviewer)


