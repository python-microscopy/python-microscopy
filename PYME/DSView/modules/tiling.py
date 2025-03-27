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
from PYME.IO import MetaDataHandler
import numpy as np

from ._base import Plugin
class Tiler(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        dsviewer.AddMenuItem('Processing', "&Tiling", self.OnTile)
        dsviewer.AddMenuItem('Processing', "&Tiling (with cross-correlation)", lambda e: self.OnTile(correlate=True))

    def reconstruct_tiles_events(self, event=None, correlate=False):
        from PYME.Analysis import deTile
        from PYME.DSView import View3D

        dlg = wx.TextEntryDialog(self.dsviewer, 'Please enter backlash correction parameters (x, y)', 'Backlash Correction Parameters', '0.0, 0.0')
        if dlg.ShowModal():
            x_corr, y_corr = [float(x) for x in dlg.GetValue().split(',')]
        else:
            return

        #get time stamps for the middle of each frame
        t = piecewiseMapping.frames_to_times(np.arange(self.image.data.shape[2]), self.image.events, self.image.mdh) + self.image.mdh['Camera.IntegrationTime'] / 2.0
        
        st = self.image.mdh.getEntry('StartTime')
        
        x0 = self.image.mdh.getEntry('Positioning.x')
        xm = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.image.events, self.image.mdh, st, x0, b'ScannerXPos', 0, x_corr, in_frames=False)
        xps = xm(t)

        y0 = self.image.mdh.getEntry('Positioning.y')
        ym = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.image.events, self.image.mdh, st, y0, b'ScannerYPos', 0, y_corr, in_frames=False)
        yps = ym(t)

       
        
        print('xm.yvals', xm.yvals)
        print('xm.xvals', xm.xvals-st)
        print('t', t-st)
        #print('yps', yps)

        print('xps', xps)

        #dark = deTile.genDark(self.vp.do.ds, self.image.mdh)
        dark = self.image.mdh.getEntry('Camera.ADOffset')
        #flat = deTile.guessFlat(self.image.data, self.image.mdh, dark)
        flat = None
        #flat = numpy.load('d:/dbad004/23_7_flat.npy')
        #flat = flat.reshape(list(flat.shape[:2]) + [1,])

        #print dark.shape, flat.shape

        split = False

        dt = deTile.assemble_tiles(self.image.data, xps, yps, self.image.mdh, split=split, skipMoveFrames=False, dark=dark, flat=flat, correlate=correlate)#, mixmatrix = [[.3, .7], [.7, .3]])

        mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)        
        
        if dt.ndim > 2:
            View3D([dt[:,:,0][:,:,None], dt[:,:,1][:,:,None]], 'Tiled Image', mdh = mdh,parent=wx.GetTopLevelParent(self.dsviewer))
        else:
            View3D(dt, 'Tiled Image', mdh = mdh, parent=wx.GetTopLevelParent(self.dsviewer))

    def reconstruct_tiles_deterministic(self, correlate=False):
        from PYME.Analysis import deTile
        from PYME.DSView import View3D

        dark = self.image.mdh.getEntry('Camera.ADOffset')
        out = deTile.assemble_tiles_xyztc(self.image.data_xyztc, self.image.mdh, dark=dark, correlate=correlate)

        mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)
        View3D(out, 'Tiled Image', mdh=mdh, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnTile(self, event=None, correlate=False):
        if 'Tiling.XPositions' in self.image.mdh:
            self.reconstruct_tiles_deterministic(correlate)
        else:
            self.reconstruct_tiles_events(event, correlate)


    def OnTilePyramid(self, event=None):
        from PYME.Analysis import deTile
        from PYME.DSView import View3D

        x0 = self.image.mdh.getEntry('Positioning.x')
        xm = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.image.events, self.image.mdh,
                                                                  self.image.mdh.getEntry('StartTime'), x0,
                                                                  b'ScannerXPos', 0, .0055)

        y0 = self.image.mdh.getEntry('Positioning.y')
        ym = piecewiseMapping.GenerateBacklashCorrPMFromEventList(self.image.events, self.image.mdh,
                                                                  self.image.mdh.getEntry('StartTime'), y0,
                                                                  b'ScannerYPos', 0, .0035)

        #dark = deTile.genDark(self.vp.do.ds, self.image.mdh)
        dark = self.image.mdh.getEntry('Camera.ADOffset')
        #flat = deTile.guessFlat(self.image.data, self.image.mdh, dark)
        flat = None
        #flat = numpy.load('d:/dbad004/23_7_flat.npy')
        #flat = flat.reshape(list(flat.shape[:2]) + [1,])

        #print dark.shape, flat.shape

        split = False

        dt = deTile.tile_pyramid('/Users/david/pyramid/', self.image.data, xm, ym, self.image.mdh, split=split, skipMoveFrames=False, dark=dark,
                         flat=flat)#, mixmatrix = [[.3, .7], [.7, .3]])

        #mdh = MetaDataHandler.NestedClassMDHandler(self.image.mdh)

        #if dt.ndim > 2:
        #    View3D([dt[:, :, 0][:, :, None], dt[:, :, 1][:, :, None]], 'Tiled Image', mdh=mdh,
        #           parent=wx.GetTopLevelParent(self.dsviewer))
        #else:
        #    View3D(dt, 'Tiled Image', mdh=mdh, parent=wx.GetTopLevelParent(self.dsviewer))

        return dt


def Plug(dsviewer):
    return Tiler(dsviewer)

    
