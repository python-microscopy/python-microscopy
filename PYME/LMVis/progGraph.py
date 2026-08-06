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
                self.subplot1 = self.figure.add_subplot(111)
                self.subplot2 = self.subplot1.twinx()

            a, ed = numpy.histogram(self.fitResults['tIndex'], int(self.Size[0]/2))

            self.subplot1.cla()
            self.subplot1.plot(ed[:-1], a/numpy.diff(ed[:2]), color='b', label='Rate' )
            self.subplot1.set_xticks([0, ed.max()])
            self.subplot1.set_yticks([0, numpy.floor(a.max()/numpy.diff(ed[:2]))[0]])
            self.subplot1.set_xlabel('Frame')
            self.subplot1.set_ylabel('#/Frame', color='b')
            self.subplot2.cla()
            self.subplot2.yaxis.set_label_position('right')
            #cs =
            self.subplot2.plot(ed[:-1], numpy.cumsum(a), color='g', label='Cumulative' )
            self.subplot2.set_xticks([0, ed.max()])
            self.subplot2.set_yticks([0, a.sum()])
            self.subplot2.set_ylabel('Total', color='g')
            self.figure.tight_layout()
            self.canvas.draw()

