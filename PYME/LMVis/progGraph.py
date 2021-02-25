#!/usr/bin/python

##################
# progGraph.py
#
# Copyright David Baddeley, 2009
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

from PYME.contrib import wxPlotPanel
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
                #self.subplot1 = self.figure.add_axes([.14,.55,.85,.44])#self.figure.add_subplot( 211 )
                #self.subplot2 = self.figure.add_axes([.14,.05,.85,.44])#self.figure.add_subplot( 212 )
                
                self.subplot1 = self.figure.add_axes([.14,.05,.85,.9])#self.figure.add_subplot( 211 )
                self.subplot2 = self.subplot1.twinx()#self.figure.add_subplot( 212 )

            a, ed = numpy.histogram(self.fitResults['tIndex'], int(self.Size[0]/2))
            print((float(numpy.diff(ed[:2]))))

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

