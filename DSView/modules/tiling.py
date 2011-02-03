#!/usr/bin/python
##################
# tiling.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
from PYME.Acquire.mytimer import mytimer

class tiler:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.dataSource = dsviewer.dataSource
        self.mdh = dsviewer.mdh
        self.ds = dsviewer.ds

        EXTRAS_TILE = wx.NewId()
        dsviewer.mExtras.Append(EXTRAS_TILE, "&Tiling", "", wx.ITEM_NORMAL)
        wx.EVT_MENU(dsviewer, EXTRAS_TILE, self.OnTile)

    def OnTile(self, event):
        from PYME.Analysis import deTile
        from PYME.DSView.dsviewer_npy import View3D

        x0 = self.mdh.getEntry('Positioning.Stage_X')
        xm = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.elv.eventSource, self.mdh, self.mdh.getEntry('StartTime'), x0, 'ScannerXPos', 0, .0055)

        y0 = self.mdh.getEntry('Positioning.Stage_Y')
        ym = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.elv.eventSource, self.mdh, self.mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0, .0035)

        #dark = deTile.genDark(self.vp.do.ds, self.mdh)
        dark = self.mdh.getEntry('Camera.ADOffset')
        flat = deTile.guessFlat(self.vp.do.ds, self.mdh, dark)
        #flat = numpy.load('d:/dbad004/23_7_flat.npy')
        #flat = flat.reshape(list(flat.shape[:2]) + [1,])

        #print dark.shape, flat.shape

        split = False

        dt = deTile.tile(self.vp.do.ds, xm, ym, self.mdh, split=split, skipMoveFrames=False, dark=dark, flat=flat)#, mixmatrix = [[.3, .7], [.7, .3]])
        if dt.ndim > 2:
            View3D([dt[:,:,0][:,:,None], dt[:,:,1][:,:,None]], 'Tiled Image')
        else:
            View3D(dt, 'Tiled Image')

    
