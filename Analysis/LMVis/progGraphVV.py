#!/usr/bin/python

##################
# progGraphVV.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#from PYME.misc import wxPlotPanel
import visvis as vv
import wx
import numpy

vv.use('wx')

class progPanel(wx.Panel):
    def __init__(self, parent, fitResults, **kwargs ):
        self.fitResults = fitResults

        wx.Panel.__init__( self, parent, **kwargs )

        self.SetMinSize((-1, 200))

        #self.panel = wx.Panel(self)

        # Make figure using "self" as a parent
        self.fig = vv.backends.backend_wx.Figure(self)

        # Make sizer and embed stuff
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        #self.sizer.Add(self.panel, 1, wx.EXPAND)
        self.sizer.Add(self.fig._widget, 2, wx.EXPAND)

        # Make callback
        #but.Bind(wx.EVT_BUTTON, self._Plot)
        #self.Bind(wx.EVT_SIZE, self._onSize)

        # Apply sizers
        self.SetSizer(self.sizer)
        self.SetAutoLayout(True)
        self.Layout()

    def _onSize( self, event ):
        #self._resizeflag = True
        self.draw()

    def draw( self ):
            """Draw data."""
            if len(self.fitResults) == 0:
                return
            
            # Make sure our figure is the active one
            vv.figure(self.fig.nr)
            
            if not hasattr( self, 'subplot1' ):
                self.subplot1 = vv.subplot(211)
                #self.subplot1.position = (30, 2, -32, -32)
                self.subplot2 = vv.subplot(212)
                #self.subplot1.position = (30, 2, -32, -32)

            

            a, ed = numpy.histogram(self.fitResults['tIndex'], self.Size[0]/6)
            print float(numpy.diff(ed[:2]))

            self.subplot1.MakeCurrent()
            vv.cla()
            vv.plot(ed[:-1], a/float(numpy.diff(ed[:2])), lc='b', lw=2)
            #self.subplot1.set_xticks([0, ed.max()])
            #self.subplot1.set_yticks([0, numpy.floor(a.max()/float(numpy.diff(ed[:2])))])
            self.subplot2.MakeCurrent()
            vv.cla()
            #cs =
            csa = numpy.cumsum(a)
            vv.plot(ed[:-1], csa/float(csa[-1]), lc='g', lw=2)
            #self.subplot2.set_xticks([0, ed.max()])
            #self.subplot2.set_yticks([0, a.sum()])

            self.fig.DrawNow()
            self.subplot1.position = (20, 2, -22, -32)
            self.subplot2.position = (20, 2, -22, -32)

