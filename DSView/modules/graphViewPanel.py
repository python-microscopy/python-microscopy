#!/usr/bin/python
##################
# graphViewPanel.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import wx
import numpy as np

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.figure import Figure

from PYME.DSView.displayOptions import DisplayOpts


class GraphViewPanel(wx.Panel):
    def __init__(self, parent, dstack = None, do = None, xvals=None, xlabel=''):
        wx.Panel.__init__(self, parent)

        if (dstack == None and do == None):
            dstack = scipy.zeros((10,10))

        if do == None:
            self.do = DisplayOpts(dstack, aspect=aspect)
            self.do.Optimise()
        else:
            self.do = do

        self.do.WantChangeNotification.append(self.draw)

        self.xvals = xvals
        self.xlabel = xlabel
            
        sizer1 = wx.BoxSizer(wx.VERTICAL)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)

        sizer1.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)

        self.Bind(wx.EVT_SIZE, self._onSize)

        self.SetSizerAndFit(sizer1)
        self.draw()

    def draw(self, event=None):
        self.axes.cla()

        if self.xvals == None:
            xvals = np.arange(self.do.ds.shape[0])
        else:
            xvals = self.xvals

        for i in range(self.do.ds.shape[3]):
            self.axes.plot(xvals, self.do.ds[:,self.do.yp, self.do.zp, i].squeeze(), label=self.do.names[i])

        self.axes.legend()
        self.axes.set_xlabel(self.xlabel)

        self.canvas.draw()

    def _onSize( self, event ):
        #self._resizeflag = True
        self._SetSize()


    def _SetSize( self ):
        pixels = tuple( self.GetClientSize() )
        self.SetSize( pixels )
        self.canvas.SetSize( pixels )
        self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                     float( pixels[1] )/self.figure.get_dpi() )

def Plug(dsviewer):
    if 'xvalues' in dsviewer.image.mdh.getEntryNames():
        xvals = dsviewer.image.mdh.getEntry('xvalues')
        xlabel = dsviewer.image.mdh.getEntry('xlabel')

    else:
        xvals = None
        xlabel = ''
        
    dsviewer.gv = GraphViewPanel(dsviewer, do=dsviewer.do, xvals=xvals, xlabel=xlabel)
    dsviewer.AddPage(dsviewer.gv, True, 'Graph View')