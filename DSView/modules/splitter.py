# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 14:48:07 2011

@author: dbad004
"""

import wx
#from PYME.Acquire.Hardware import splitter
from PYME.Analysis.DataSources import UnsplitDataSource
import numpy as np
#from PYME.DSView.arrayViewPanel import *
                                       
    
class Unmixer:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image 
        

        EXTRAS_UNMUX = wx.NewId()
        dsviewer.mProcessing.Append(EXTRAS_UNMUX, "&Unsplit", "", wx.ITEM_NORMAL)
        wx.EVT_MENU(dsviewer, EXTRAS_UNMUX, self.OnUnmix)

        EXTRAS_SETSF = wx.NewId()
        dsviewer.mProcessing.Append(EXTRAS_SETSF, "Set Shift Field", "", wx.ITEM_NORMAL)
        wx.EVT_MENU(dsviewer, EXTRAS_SETSF, self.OnSetShiftField)

    def OnUnmix(self, event):
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack

        mdh = self.image.mdh
        if 'chroma.dx' in mdh.getEntryNames():
            sf = (mdh['chroma.dx'], mdh['chroma.dy'])
        else:
            sf = None

        flip = True
        if 'Splitter.Flip' in mdh.getEntryNames() and not mdh['Splitter.Flip']:
            flip = False

        ROIX1 = mdh.getEntry('Camera.ROIPosX')
        ROIY1 = mdh.getEntry('Camera.ROIPosY')

        ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
        ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

        um0 = UnsplitDataSource.DataSource(self.image.data,
                                           [ROIX1, ROIY1, ROIX2, ROIY2],
                                           0, flip, sf)

        um1 = UnsplitDataSource.DataSource(self.image.data, 
                                           [ROIX1, ROIY1, ROIX2, ROIY2], 1
                                           , flip, sf)
            
        im = ImageStack([um0, um1], titleStub = 'Unmixed Image')
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        #im.mdh['Processing.GaussianFilter'] = sigmas

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnSetShiftField(self, event=None):
        from PYME.FileUtils import nameUtils
        fdialog = wx.FileDialog(None, 'Please select shift field to use ...',
                    wildcard='Shift fields|*.sf', style=wx.OPEN, defaultDir = nameUtils.genShiftFieldDirectoryPath())
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            #self.ds = example.CDataStack(fdialog.GetPath().encode())
            #self.ds =
            sfFilename = fdialog.GetPath()
            self.image.mdh.setEntry('chroma.ShiftFilename', sfFilename)
            dx, dy = np.load(sfFilename)
            self.image.mdh.setEntry('chroma.dx', dx)
            self.image.mdh.setEntry('chroma.dy', dy)
            #self.md.setEntry('PSFFile', psfFilename)
            #self.stShiftFieldName.SetLabel('Shifts: %s' % os.path.split(sfFilename)[1])
            #self.stShiftFieldName.SetForegroundColour(wx.Colour(0, 128, 0))
            return True
        else:
            return False


def Plug(dsviewer):
    dsviewer.unmux = Unmixer(dsviewer)
                                       
    