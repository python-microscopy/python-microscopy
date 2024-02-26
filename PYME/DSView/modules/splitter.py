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
from PYME.IO.DataSources import UnsplitDataSource
import numpy as np
#from PYME.DSView.arrayViewPanel import *

import logging
logger=logging.getLogger(__name__)
        
global_shiftfield = None                               

from ._base import Plugin
class Unmixer(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self,dsviewer)
        
        dsviewer.AddMenuItem('Processing', "&Unsplit\tCtrl-U", self.OnUnmix)
        dsviewer.AddMenuItem('Processing', "&Unsplit, taking brightest\tCtrl-Shift-U", self.OnUnmixMax)
        dsviewer.AddMenuItem('Processing', "Set Shift Field", self.OnSetShiftField)

    def _getUSDataSources(self):
        from PYME.Analysis import splitting

        mdh = self.image.mdh

        si = splitting.SplittingInfo(self.image.mdh, self.image.shape[:2])
        
        if 'chroma.dx' in mdh.getEntryNames():
            sf = (mdh['chroma.dx'], mdh['chroma.dy'])
        elif global_shiftfield:
            sf = global_shiftfield
        else:
            sf = None


        usds = [UnsplitDataSource.DataSource(self.image.data, si, i, sf, voxelsize=self.image.voxelsize) for i in
                range(si.num_chans)]

        return usds

    def OnUnmix(self, event):
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack

        usds = self._getUSDataSources()

            
        fns = os.path.split(self.image.filename)[1]
        im = ImageStack(usds, titleStub = '%s - unsplit' % fns)
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename
        # grab events from the first data source TODO - do this need to be smarter?
        try:
            im.events = usds[0].getEvents()
        except:
            logger.warning('No Events found when coalescing UnsplitDataSource')

        if 'fitResults' in dir(self.image):
            im.fitResults = self.image.fitResults
        #im.mdh['Processing.GaussianFilter'] = sigmas

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'
            
        #print(im.data[:,:,0,1].shape)

        dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnUnmixMax(self, event):
        #unmix and take brightest channel
        #from PYME.Analysis import deTile
        from PYME.DSView import ViewIm3D, ImageStack

        usds = self._getUSDataSources()
            
        fns = os.path.split(self.image.filename)[1]

        zm = int(usds[0].shape[2]/2)

        maxs = [u[:,:,zm].max() for u in usds]
        im = ImageStack(usds[np.argmax(maxs)], titleStub = '%s - unsplit' % fns)

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
        from PYME.IO.FileUtils import nameUtils
        fdialog = wx.FileDialog(None, 'Please select shift field to use ...',
                    wildcard='Shift fields|*.sf', style=wx.FD_OPEN, defaultDir = nameUtils.genShiftFieldDirectoryPath())
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
    return Unmixer(dsviewer)
                                       
    
