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

from scipy import ndimage
import numpy as np

from PYME.DSView.dsviewer import ViewIm3D, ImageStack
from ._base import Plugin
class MultiThreshold(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self.dsviewer)

        
        dsviewer.AddMenuItem('Processing', "Object # vs threshold", self.OnPlotProfile)
        
        
        #accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL,  ord('k'), PLOT_PROFILE )])
        #self.dsviewer.SetAcceleratorTable(accel_tbl)


    def OnPlotProfile(self, event=None):
        try:
            names = self.image.mdh.getEntry('ChannelNames')
        except:
            names = ['Channel %d' % d for d in range(self.image.data.shape[3])]

        
        #print lx, hx, ly, hy

        #pylab.figure()
        plots = []

        for chanNum in range(self.image.data.shape[3]):

            tmin = self.do.Offs[chanNum]
            tmax = tmin + 1./self.do.Gains[chanNum]
            
            d = self.image.data[:,:,:,chanNum].squeeze()

            trange = np.linspace(tmin, tmax)

            p = np.array([ndimage.label(d > t)[1] for t in trange])

            plots.append(p.reshape(-1, 1,1))

        #pylab.legend(names)

        im = ImageStack(plots, titleStub='New Threshold Range')
        im.xvals = trange

        
        im.xlabel = 'Threshold'

        im.ylabel = 'Num Objects'
        im.defaultExt = '.txt'

        #im.mdh['voxelsize.x'] = voxx
        im.mdh['ChannelNames'] = names
        
        im.mdh['Profile.XValues'] = im.xvals
        im.mdh['Profile.XLabel'] = im.xlabel
        im.mdh['Profile.YLabel'] = im.ylabel
        

        im.mdh['OriginalImage'] = self.image.filename

        ViewIm3D(im, mode='graph', parent=wx.GetTopLevelParent(self.dsviewer))




   


def Plug(dsviewer):
    return MultiThreshold(dsviewer)
    
