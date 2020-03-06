#!/usr/bin/env python

"""
New version based on the original by Edward Abraham, incorporating some
forum discussion, minor bugfixes, and working for Python 2.5.2,
wxPython 2.8.7.1, and Matplotlib 0.98.3.

I haven't found an advantage to using the NoRepaintCanvas, so it's removed.

John Bender, CWRU, 10 Sep 08
"""

import matplotlib
matplotlib.interactive( True )
#matplotlib.use( 'WXAgg' )

import numpy as num
import wx

class PlotPanel (wx.Panel):
    """The PlotPanel has a Figure and a Canvas. OnSize events simply set a
flag, and the actual resizing of the figure is triggered by an Idle event."""
    def __init__( self, parent, color=None, dpi=None, **kwargs ):
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.figure import Figure

        self.parent = parent

        # initialize Panel
        if 'id' not in kwargs.keys():
            kwargs['id'] = wx.ID_ANY
        if 'style' not in kwargs.keys():
            kwargs['style'] = wx.NO_FULL_REPAINT_ON_RESIZE
        wx.Panel.__init__( self, parent, **kwargs )

        # initialize matplotlib stuff
        self.figure = Figure( None, dpi )
        self.canvas = FigureCanvasWxAgg( self, -1, self.figure )
        self.SetColor( color )

        self._SetSize()
        #self.draw()

        self._resizeflag = False
        
        self._vis_flag = False

        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self.Bind(wx.EVT_SET_FOCUS, lambda e: self.draw())

    def SetColor( self, rgbtuple=None ):
        """Set figure and canvas colours to be the same."""
        if rgbtuple is None:
            rgbtuple = wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ).Get()
        clr = [c/255. for c in rgbtuple]
        self.figure.set_facecolor( clr )
        self.figure.set_edgecolor( clr )
        self.canvas.SetBackgroundColour( wx.Colour( *rgbtuple ) )

    def _onSize( self, event ):
        self._resizeflag = True
        #print 'Resizing Plot'
        #self._SetSize()

    def _onIdle( self, evt ):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()
            
        #print(self.IsShown())
        if self.IsShownOnScreen():
            if not self._vis_flag:
                #print('s')
                self._vis_flag = True
                self.draw()
        else:
            self._vis_flag = False

    def _SetSize( self ):
        pixels = tuple( self.GetClientSize() )
        if not tuple(self.canvas.GetSize()) == pixels:
            self.SetSize( pixels )
            self.canvas.SetSize( pixels )
            self.figure.set_size_inches( float( pixels[0] )/self.figure.get_dpi(),
                                         float( pixels[1] )/self.figure.get_dpi() )
            try:
                if self.IsShownOnScreen():
                    self.draw()
            except:
                pass

    def draw(self): pass # abstract, to be overridden by child classes

if __name__ == '__main__':
    class DemoPlotPanel (PlotPanel):
        """Plots several lines in distinct colors."""
        def __init__( self, parent, point_lists, clr_list, **kwargs ):
            self.parent = parent
            self.point_lists = point_lists
            self.clr_list = clr_list

            # initiate plotter
            PlotPanel.__init__( self, parent, **kwargs )
            self.SetColor( (255,255,255) )

        def draw( self ):
            """Draw data."""
            if not hasattr( self, 'subplot' ):
                self.subplot = self.figure.add_subplot( 111 )

            for i, pt_list in enumerate( self.point_lists ):
                plot_pts = num.array( pt_list )
                clr = [float( c )/255. for c in self.clr_list[i]]
                self.subplot.plot( plot_pts[:,0], plot_pts[:,1], color=clr )

    theta = num.arange( 0, 45*2*num.pi, 0.02 )

    rad0 = (0.8*theta/(2*num.pi) + 1)
    r0 = rad0*(8 + num.sin( theta*7 + rad0/1.8 ))
    x0 = r0*num.cos( theta )
    y0 = r0*num.sin( theta )

    rad1 = (0.8*theta/(2*num.pi) + 1)
    r1 = rad1*(6 + num.sin( theta*7 + rad1/1.9 ))
    x1 = r1*num.cos( theta )
    y1 = r1*num.sin( theta )

    points = [[(xi,yi) for xi,yi in zip( x0, y0 )],
              [(xi,yi) for xi,yi in zip( x1, y1 )]]
    clrs = [[225,200,160], [219,112,147]]

    app = wx.PySimpleApp( 0 )
    frame = wx.Frame( None, wx.ID_ANY, 'WxPython and Matplotlib', size=(300,300) )
    panel = DemoPlotPanel( frame, points, clrs )
    frame.Show()
    app.MainLoop()
