#!/usr/bin/python

##################
# progGraph.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.misc import wxPlotPanel
import numpy

class progPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, fitResults, **kwargs ):
        self.fitResults = fitResults

        wxPlotPanel.PlotPanel.__init__( self, parent, **kwargs )

    def draw( self ):
            """Draw data."""
            if len(self.fitResults) == 0:
                return
            
            if not hasattr( self, 'subplot1' ):
                self.subplot1 = self.figure.add_axes([.14,.55,.85,.44])#self.figure.add_subplot( 211 )
                self.subplot2 = self.figure.add_axes([.14,.05,.85,.44])#self.figure.add_subplot( 212 )

            a, ed = numpy.histogram(self.fitResults['tIndex'], self.Size[0]/2)
            print float(numpy.diff(ed[:2]))

            self.subplot1.cla()
            self.subplot1.plot(ed[:-1], a/float(numpy.diff(ed[:2])), color='b' )
            self.subplot1.set_xticks([0, ed.max()])
            self.subplot1.set_yticks([0, numpy.floor(a.max()/float(numpy.diff(ed[:2])))])
            self.subplot2.cla()
            #cs =
            self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g' )
            self.subplot2.set_xticks([0, ed.max()])
            self.subplot2.set_yticks([0, a.sum()])

            self.canvas.draw()

