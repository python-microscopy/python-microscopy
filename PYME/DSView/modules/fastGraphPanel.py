#!/usr/bin/python
##################
# graphViewPanel.py
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
#import wx
#import wx.lib.agw.aui as aui
import numpy as np

from PYME.ui import fastGraph
from PYME.DSView.displayOptions import DisplayOpts



class GraphViewPanel(fastGraph.FastGraphPanel):
    def __init__(self, parent, dstack = None, do = None, xvals=None, xlabel='',ylabel=''):
        

        if (dstack is None and do is None):
            dstack = np.zeros((10,10))

        if do is None:
            self.do = DisplayOpts(dstack)
            self.do.Optimise()
        else:
            self.do = do
            
        

        self.do.WantChangeNotification.append(self.draw)
        
        if xvals is None:
            xvals = np.arange(self.do.ds.shape[0])

        self.xvals = xvals
        self.xlabel = xlabel
        
        fastGraph.FastGraphPanel.__init__(self, parent, -1, self.xvals, self.do.ds[:, :])

        #self.draw()

    def draw(self, event=None):
        self.SetData(self.xvals, self.do.ds[:, :])

   

def Plug(dsviewer):
    #if dsviewer.image.mdh and 'xvalues' in dsviewer.image.mdh.getEntryNames():
    #    xvals = dsviewer.image.mdh.getEntry('xvalues')
    #    xlabel = dsviewer.image.mdh.getEntry('xlabel')

    if 'xvals' in dir(dsviewer.image):
        xvals = dsviewer.image.xvals
        xlabel = dsviewer.image.xlabel

    else:
        xvals = None
        xlabel = ''

    if 'ylabel' in dir(dsviewer.image):
        ylabel = dsviewer.image.ylabel
    else:
        ylabel=''
        
    gv = GraphViewPanel(dsviewer, do=dsviewer.do, xvals=xvals, xlabel=xlabel, ylabel=ylabel)
    dsviewer.AddPage(gv, True, 'Graph View')
    
    return {'gv' : gv}
    
