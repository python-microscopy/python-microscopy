#!/usr/bin/python
##################
# tiling.py
#
# Copyright David Baddeley, 2011
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
##################

import wx
from PYME.Analysis import piecewiseMapping

class tiler:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        self.image = dsviewer.image

        dsviewer.AddMenuItem('Processing', "&Tiling", self.OnTile)

    def OnTile(self, event):
        from PYME.Analysis import deTile
        from PYME.DSView import View3D

        x0 = self.image.mdh.getEntry('Positioning.Stage_X')
        xm = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.image.events, self.image.mdh, self.image.mdh.getEntry('StartTime'), x0, 'ScannerXPos', 0, .0055)

        y0 = self.image.mdh.getEntry('Positioning.Stage_Y')
        ym = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.image.events, self.image.mdh, self.image.mdh.getEntry('StartTime'), y0, 'ScannerYPos', 0, .0035)

        #dark = deTile.genDark(self.vp.do.ds, self.image.mdh)
        dark = self.image.mdh.getEntry('Camera.ADOffset')
        flat = deTile.guessFlat(self.image.data, self.image.mdh, dark)
        #flat = numpy.load('d:/dbad004/23_7_flat.npy')
        #flat = flat.reshape(list(flat.shape[:2]) + [1,])

        #print dark.shape, flat.shape

        split = False

        dt = deTile.tile(self.image.data, xm, ym, self.image.mdh, split=split, skipMoveFrames=False, dark=dark, flat=flat)#, mixmatrix = [[.3, .7], [.7, .3]])
        if dt.ndim > 2:
            View3D([dt[:,:,0][:,:,None], dt[:,:,1][:,:,None]], 'Tiled Image', parent=wx.GetTopLevelParent(self.dsviewer))
        else:
            View3D(dt, 'Tiled Image', parent=wx.GetTopLevelParent(self.dsviewer))


def Plug(dsviewer):
    dsviewer.tiler = tiler(dsviewer)

    
