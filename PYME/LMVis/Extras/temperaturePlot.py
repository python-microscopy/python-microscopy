#!/usr/bin/python
##################
# temperaturePlot.py
#
# Copyright David Baddeley, 2010
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

class TempPlotter:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Extras>Diagnostics', "Plot temperature record", self.OnPlotTemperature)

    def OnPlotTemperature(self, event):
        from PYME.misc import tempDB
        # import pylab
        import matplotlib.pyplot as plt
        
        pipeline = self.visFr.pipeline

        t, tm = tempDB.getEntries(pipeline.mdh.getEntry('StartTime'), pipeline.mdh.getEntry('EndTime'))
        t_, tm_ = tempDB.getEntries(pipeline.mdh.getEntry('StartTime') - 3600, pipeline.mdh.getEntry('EndTime'))

        plt.figure()
        plt.plot((t_ - pipeline.mdh.getEntry('StartTime'))/60, tm_)
        plt.plot((t - pipeline.mdh.getEntry('StartTime'))/60, tm, lw=2)
        plt.xlabel('Time [mins]')
        plt.ylabel('Temperature [C]')



def Plug(visFr):
    """Plugs this module into the gui"""
    TempPlotter(visFr)


