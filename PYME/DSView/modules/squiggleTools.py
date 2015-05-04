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
import pylab
from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer_npy_nb import ViewIm3D, ImageStack

class squiggle:
    def __init__(self, dsviewer):
        self.dsviewer = dsviewer

        #self.view = dsviewer.view
        self.do = dsviewer.do
        self.image = dsviewer.image
        
        dsviewer.AddMenuItem('Processing', "Plot wavy profile", self.OnPlotProfile)


    def OnPlotProfile(self, event=None):
        #lx, ly, hx, hy = self.do.GetSliceSelection()
        pts = np.array(self.do.selection_trace)

        w = int(np.floor(0.5*self.do.selectionWidth))

        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % d for d in range(self.image.data.shape[3])]

        try:
            voxx = self.image.mdh.getEntry('voxelsize.x')
        except:
            voxx=1

        plots = []
        t = np.arange(np.ceil(len(pts)))

        for chanNum in range(self.image.data.shape[3]):       
            p = ndimage.map_coordinates(self.image.data[:,:, self.do.zp, chanNum].squeeze(), pts.T)

            plots.append(p.reshape(-1, 1,1))

        #pylab.legend(names)

        im = ImageStack(plots, titleStub='New Profile')
        im.xvals = t*voxx

        if not voxx == 1:
            im.xlabel = 'Distance [um]'
        else:
            im.xlabel = 'Distance [pixels]'

        im.ylabel = 'Intensity'
        im.defaultExt = '.txt'

        im.mdh['voxelsize.x'] = voxx
        im.mdh['ChannelNames'] = names
        im.mdh['Profile.XValues'] = im.xvals
        im.mdh['Profile.XLabel'] = im.xlabel
        im.mdh['Profile.YLabel'] = im.ylabel
        #im.mdh['Profile.StartX'] = lx
        #im.mdh['Profile.StartY'] = ly
        #im.mdh['Profile.EndX'] = hx
        #im.mdh['Profile.EndY'] = hy
        #im.mdh['Profile.Width'] = 2*w + 1

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph')













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


    def OnZPlotProfile(self, event):
        x,p,d, pi = self.vp.GetProfile(50, background=[7,7])

        pylab.figure(1)
        pylab.clf()
        pylab.step(x,p)
        pylab.step(x, 10*d - 30)
        pylab.ylim(-35,pylab.ylim()[1])

        pylab.xlim(x.min(), x.max())

        pylab.xlabel('Time [%3.2f ms frames]' % (1e3*self.mdh.getEntry('Camera.CycleTime')))
        pylab.ylabel('Intensity [counts]')

        fr = self.fitResults[pi]

        if not len(fr) == 0:
            pylab.figure(2)
            pylab.clf()

            pylab.subplot(211)
            pylab.errorbar(fr['tIndex'], fr['fitResults']['x0'] - self.vp.do.xp*1e3*self.mdh.getEntry('voxelsize.x'), fr['fitError']['x0'], fmt='xb')
            pylab.xlim(x.min(), x.max())
            pylab.xlabel('Time [%3.2f ms frames]' % (1e3*self.mdh.getEntry('Camera.CycleTime')))
            pylab.ylabel('x offset [nm]')

            pylab.subplot(212)
            pylab.errorbar(fr['tIndex'], fr['fitResults']['y0'] - self.vp.do.yp*1e3*self.mdh.getEntry('voxelsize.y'), fr['fitError']['y0'], fmt='xg')
            pylab.xlim(x.min(), x.max())
            pylab.xlabel('Time [%3.2f ms frames]' % (1e3*self.mdh.getEntry('Camera.CycleTime')))
            pylab.ylabel('y offset [nm]')

            pylab.figure(3)
            pylab.clf()

            pylab.errorbar(fr['fitResults']['x0'] - self.vp.do.xp*1e3*self.mdh.getEntry('voxelsize.x'),fr['fitResults']['y0'] - self.vp.do.yp*1e3*self.mdh.getEntry('voxelsize.y'), fr['fitError']['x0'], fr['fitError']['y0'], fmt='xb')
            #pylab.xlim(x.min(), x.max())
            pylab.xlabel('x offset [nm]')
            pylab.ylabel('y offset [nm]')


def Plug(dsviewer):
    dsviewer.squiggle = squiggle(dsviewer)
    
