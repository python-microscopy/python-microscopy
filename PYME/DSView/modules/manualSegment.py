#!/usr/bin/python
##################
# profilePlotting.py
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

#from PYME.Acquire.mytimer import mytimer
# import pylab
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer import ViewIm3D, ImageStack

from ._base import Plugin
class ManualSegment(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        self.mask = None
        self.rois = []
        
        dsviewer.do.overlays.append(self.DrawOverlays)
        
        dsviewer.AddMenuItem('Segmentation', 'Create mask', self.OnCreateMask)
        dsviewer.AddMenuItem('Segmentation', "Fill selection\tCtrl-F", self.FillSelection)
        dsviewer.AddMenuItem('Segmentation', 'Multiply image with mask', self.OnMultiplyMask)

        #accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('k'), PLOT_PROFILE )])
        #self.dsviewer.SetAcceleratorTable(accel_tbl)
        
    def OnMultiplyMask(self, event=None):
        new_image = []
        
        mask = self.mask > 0.5 #generate a binary mask from labels
        
        for chNum in range(self.image.data.shape[3]):
            new_image.append(mask*self.image.data[:,:,:,chNum])

        im = ImageStack(new_image, titleStub='Masked image')
        im.mdh.copyEntriesFrom(self.image.mdh)
            
        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    def OnCreateMask(self, event=None):
        #lx, ly, hx, hy = self.do.GetSliceSelection()
        import numpy as np
        #from scipy import ndimage
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D

        #sp = self.image.data.shape[:3]
        #if len(sp)        
        self.mask = np.atleast_3d(np.zeros(self.image.data.shape[:3], 'uint16'))
        self.vmax = 0
        self.image.labels = self.mask
        
        im = ImageStack(self.mask, titleStub = 'Manual labels')
        im.mdh.copyEntriesFrom(self.image.mdh)

        #im.mdh['Processing.CropROI'] = roi

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
        
        self.rois = []

        #set scaling to (0,1)
        for i in range(im.data.shape[3]):
            self.dv.do.Gains[i] = 1.0
            
    def OnKeyPress(self, event):
        key = event.GetKeyCode()
        
        if key == ord('F'):
            self.FillSelection()
        
        event.Skip()
        
    def FillSelection(self, event=None):
        if self.mask is None:
            print('Create a mask first')
            return
        
#        import Image, ImageDraw
#        import pylab
#        x, y = np.array(self.do.selection_trace).T
#        x0 = x.min()
#        x1 =  x.max()
#        y0 = y.min()
#        y1 = y.max()
#        
#        w = x1 - x0
#        h = y1 - y0
#        
#        img = Image.new('L', (h, w))
#        ImageDraw.Draw(img).polygon(np.hstack([y - y0,x-x0]).T, outline=1, fill=1)
#        pylab.imshow(np.array(img))
#        self.mask[x0:x1, y0:y1, self.do.zp] = np.array(img)
        
        from skimage import draw
        from shapely.geometry import Polygon, Point
        p = Polygon(self.do.selection_trace)
        #x0, y0, x1, y1 = p.bounds

        self.vmax += 1
        rr, cc = draw.polygon(*np.array(self.do.selection_trace).T)
        self.mask[rr,cc, self.do.zp] = self.vmax
        
        # for x in range(int(x0), int(x1)+1):
        #     for y in range (int(y0), int(y1)+1):
        #         if Point(x, y).within(p):
        #             self.mask[x,y,self.do.zp] = self.vmax
        self.rois.append(p)
        self.dv.update()
        self.dv.Refresh()
    
        #pylab.legend(names)
        
    def DrawOverlays(self, view, dc):
        import wx
        col = wx.TheColourDatabase.FindColour('WHITE')
        dc.SetPen(wx.Pen(col,1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        
        for roi in self.rois:
            x, y = roi.exterior.xy
            pts = np.vstack(view._PixelToScreenCoordinates(np.array(x), np.array(y))).T
            dc.DrawSpline(pts)
            
        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)


        #dsviewer.paneHooks.append(self.GenProfilePanel)

#    def GenProfilePanel(self, _pnl):
#        item = afp.foldingPane(_pnl, -1, caption="Intensity Profile", pinned = True)
##        item = self._pnl.AddFoldPanel("Intensity Profile", collapsed=False,
##                                      foldIcons=self.Images)
#
#        bPlotProfile = wx.Button(item, -1, 'Plot')
#
#        bPlotProfile.Bind(wx.EVT_BUTTON, self.OnPlotProfile)
#        #self._pnl.AddFoldPanelWindow(item, bPlotProfile, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)
#        item.AddNewElement(bPlotProfile)
#        _pnl.AddPane(item)


   


def Plug(dsviewer):
    mSegment = ManualSegment(dsviewer)
    mSegment.OnCreateMask(None)
    
    return mSegment
    
