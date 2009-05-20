#from PYME.misc import wxPlotPanel
import wx
from enthought.chaco.api import HPlotContainer, create_line_plot
from enthought.enable.wx_backend.api import Window

import numpy

class progPanel(wx.Panel):
    def __init__(self, parent, fitResults, **kwargs ):
        self.fitResults = fitResults

        wx.Panel.__init__( self, parent, **kwargs )

        self.plot_window = Window(self, component=self.draw())
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.plot_window.control, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.SetAutoLayout(True)


    def draw( self ):
        """Draw data."""
        if len(self.fitResults) == 0:
            return

        #if not hasattr( self, 'subplot1' ):
        #    self.subplot1 = self.figure.add_subplot( 211 )
        #    self.subplot2 = self.figure.add_subplot( 212 )

        a, ed = numpy.histogram(self.fitResults['tIndex'], self.Size[0]/2)

        #self.subplot1.cla()
        #self.subplot1.plot(ed[:-1], a, color='b' )
        #self.subplot1.set_xticks([0, ed.max()])
        #self.subplot1.set_yticks([0, a.max()])
        #self.subplot2.cla()
        #self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
        #self.subplot2.set_xticks([0, ed.max()])
        #self.subplot2.set_yticks([0, a.sum()])

        plot1 = create_line_plot(ed[:,-1], a, color = 'blue', bgcolor="white",
                                add_grid=True, add_axis=True)

        plot2 = create_line_plot(ed[:,-1], numpy.cumsum(a), color = 'green', bgcolor="white",
                                add_grid=True, add_axis=True)
        container = HPlotContainer(spacing=20, padding=50, bgcolor="lightgray")
        container.add(plot1)
        container.add(plot2)
        return container


