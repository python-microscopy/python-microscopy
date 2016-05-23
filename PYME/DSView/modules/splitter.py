#!/usr/bin/python

###############
# splitter.py
#
# Copyright David Baddeley, 2012
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
################


import wx
import os
#from PYME.Acquire.Hardware import splitter
from PYME.io.DataSources import UnsplitDataSource
import numpy as np
#from PYME.DSView.arrayViewPanel import *
        
global_shiftfield = None                               
    
class Unmixer:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image 
        
        dsviewer.AddMenuItem('Processing', "&Unsplit\tCtrl-U", self.OnUnmix)
        dsviewer.AddMenuItem('Processing', "&Unsplit, taking brightest\tCtrl-Shift-U", self.OnUnmixMax)
        dsviewer.AddMenuItem('Processing', "Set Shift Field", self.OnSetShiftField)

    def OnUnmix(self, event):
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack

        mdh = self.image.mdh
        if 'chroma.dx' in mdh.getEntryNames():
            sf = (mdh['chroma.dx'], mdh['chroma.dy'])
        elif global_shiftfield:
            sf = global_shiftfield
        else:
            sf = None

        flip = True
        if 'Splitter.Flip' in mdh.getEntryNames() and not mdh['Splitter.Flip']:
            flip = False
            
        chanROIs = None
        if 'Splitter.Channel0ROI' in mdh.getEntryNames():
            chanROIs = [mdh['Splitter.Channel0ROI'],mdh['Splitter.Channel1ROI']]

        ROIX1 = mdh.getEntry('Camera.ROIPosX')
        ROIY1 = mdh.getEntry('Camera.ROIPosY')

        ROIX2 = ROIX1 + mdh.getEntry('Camera.ROIWidth')
        ROIY2 = ROIY1 + mdh.getEntry('Camera.ROIHeight')

        um0 = UnsplitDataSource.DataSource(self.image.data,
                                           [ROIX1, ROIY1, ROIX2, ROIY2],
                                           0, flip, sf, chanROIs=chanROIs, voxelsize=self.image.voxelsize)

        um1 = UnsplitDataSource.DataSource(self.image.data, 
                                           [ROIX1, ROIY1, ROIX2, ROIY2], 1
                                           , flip, sf, chanROIs=chanROIs, voxelsize=self.image.voxelsize)
            
        fns = os.path.split(self.image.filename)[1]
        im = ImageStack([um0, um1], titleStub = '%s - unsplit' % fns)
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        
        if 'fitResults' in dir(self.image):
            im.fitResults = self.image.fitResults
        #im.mdh['Processing.GaussianFilter'] = sigmas

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'
            
        #print im.data[:,:,1,1].shape

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnUnmixMax(self, event):
        #unmix and take brightest channel
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack

        mdh = self.image.mdh
        if 'chroma.dx' in mdh.getEntryNames():
            sf = (mdh['chroma.dx'], mdh['chroma.dy'])
        elif global_shiftfield:
            sf = global_shiftfield
        else:
            #self.OnSetShiftField()
            #sf = (mdh['chroma.dx'], mdh['chroma.dy'])
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
            
        fns = os.path.split(self.image.filename)[1]
        zm = um0.shape[2]/2
        if um0[:,:,zm].max() > um1[:,:,zm].max():
            im = ImageStack(um0, titleStub = '%s - unsplit' % fns)
        else:
            im = ImageStack(um1, titleStub = '%s - unsplit' % fns)
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        
        if 'fitResults' in dir(self.image):
            im.fitResults = self.image.fitResults
        #im.mdh['Processing.GaussianFilter'] = sigmas

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))


    def OnSetShiftField(self, event=None):
        from PYME.io.FileUtils import nameUtils
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
                                       
    
