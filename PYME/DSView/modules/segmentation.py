#!/usr/bin/python
##################
# blobFinding.py
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
import wx.lib.agw.aui as aui
import numpy as np
#from PYME.Acquire.mytimer import mytimer
#from scipy import ndimage
from PYME.IO.image import ImageStack
from PYME.DSView import ViewIm3D
#import time
from PYME.Analysis import annealThresh
from PYME.ui import histLimits

class SegmentationPanel(wx.Panel):
    def __init__(self, parent, sourceImage, destImage):
        wx.Panel.__init__(self, parent)
        self.cancelled = False

        self.parent = parent

        self.sourceImage = sourceImage
        self.destImage = destImage

        self.nChans = self.sourceImage.data.shape[3]

        self.numIters = 10

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        dispSize = (120, 80)

        
        self.hI50 = []
        self.tISlopes = []
        self.tNeighFracs = []
        self.tNSlopes = []

        for i in range(self.nChans):
            ssizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, self.sourceImage.names[i]), wx.VERTICAL)
            ssizer.Add(wx.StaticText(self, -1, 'Intensity for p = .5:'), 0, wx.ALL, 2)

            id = wx.NewIdRef()
            #self.hIds.append(id)
            c = self.sourceImage.data[:,:,:, i].ravel()
            #print c.min(), c.max()
            hClim = histLimits.HistLimitPanel(self, id, c[::max(1, len(c)/1e4)], c.min(), c.max(), size=dispSize, log=True, threshMode=True)
            #hClim.SetThresholdMode(True)

            #hClim.Bind(histLimits.EVT_LIMIT_CHANGE, self.OnCLimChanged)
            self.hI50.append(hClim)

            ssizer.Add(hClim, 0, wx.ALL, 2)

            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'Slope:'), 0, wx.ALL, 2)

            tSlope = wx.TextCtrl(self, -1, '1.0', size=(30, -1))
            self.tISlopes.append(tSlope)
            hsizer.Add(tSlope, 1, wx.ALL, 2)
            ssizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'N Frac:'), 0, wx.ALL, 2)

            tSlope = wx.TextCtrl(self, -1, '0.5', size=(30, -1))
            self.tNeighFracs.append(tSlope)
            hsizer.Add(tSlope, 1, wx.ALL, 2)
            ssizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)

            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            hsizer.Add(wx.StaticText(self, -1, 'N Slope:'), 0, wx.ALL, 2)

            tSlope = wx.TextCtrl(self, -1, '1', size=(30, -1))
            self.tNSlopes.append(tSlope)
            hsizer.Add(tSlope, 1, wx.ALL, 2)
            ssizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 0)
            
            sizer1.Add(ssizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, '# Iters:'), 0, wx.ALL, 2)

        self.tNIters = wx.TextCtrl(self, -1, '10', size=(30, -1))
        hsizer.Add(self.tNIters, 1, wx.ALL, 2)
        sizer1.Add(hsizer, 0, wx.ALL|wx.EXPAND, 2)

#        self.gProgress = wx.Gauge(self, -1, numIters)
#
#        sizer1.Add(self.gProgress, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)

        btn = wx.Button(self, -1, 'Apply')
        btn.Bind(wx.EVT_BUTTON, self.OnApply)

        sizer1.Add(btn, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

#        btn = wx.Button(self, -1, 'Done')
#        btn.Bind(wx.EVT_BUTTON, self.OnDone)
#
#        sizer1.Add(btn, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def OnDone(self, event):
        self.cancelled = True
        #self.EndModal(wx.ID_CANCEL)

    def _getI50(self, chan):
        return np.mean(self.hI50[chan].GetValue())

    def _getISlope(self, chan):
        return float(self.tISlopes[chan].GetValue())
    
    def _getN50(self, chan):
        return float(self.tNeighFracs[chan].GetValue())

    def _getNSlope(self, chan):
        return float(self.tNSlopes[chan].GetValue())

    def _getNIters(self):
        return int(self.tNIters.GetValue())

    def OnApply(self, event):
        for i in range(self.nChans):
            annealThresh.annealThresh2(np.atleast_3d(self.sourceImage.data[:,:,:,i].squeeze()), self._getI50(i),
                self._getISlope(i), self._getN50(i), self._getNSlope(i), nIters=self._getNIters(), out = self.destImage[i])

        self.parent.do.Optimise()
        

    def Tick(self, dec):
        if not self.cancelled:
            self.gProgress.SetValue(dec.loopcount)
            return True
        else:
            return False

from ._base import Plugin
class Segmenter(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        dsviewer.AddMenuItem('Processing', "MC Annealing Segmentation", self.OnSegmentAnneal)
        
        #dsviewer.updateHooks.append(self.update)
        
    
    def OnSegmentAnneal(self, event):
        newImages = [np.zeros(self.image.data.shape[:3], 'b') for i in range(self.image.data.shape[3])]

        im = ImageStack(newImages)
        im.mdh.copyEntriesFrom(self.image.mdh)
        im.mdh['Parent'] = self.image.filename

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.res = ViewIm3D(im, parent=wx.GetTopLevelParent(self.dsviewer), mode = mode, glCanvas = self.dsviewer.glCanvas)

        self.panAnneal = SegmentationPanel(self.res, self.image, newImages)

        self.pinfo1 = aui.AuiPaneInfo().Name("annealPanel").Left().Caption('Segmentation').DestroyOnClose(True).CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        self.res._mgr.AddPane(self.panAnneal, self.pinfo1)
        self.res._mgr.Update()


    #def update(self):
    #    if 'decvp' in dir(self):
    #        self.decvp.imagepanel.Refresh()

def Plug(dsviewer):
    return Segmenter(dsviewer)
