
##################
# multiviewMapping.py
#
# Copyright Andrew Barentine, David Baddeley
# david.baddeley@yale.edu
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

import numpy as np
import wx
from PYME.Analysis.points.DeClump import pyDeClump

class clumper:
    """


    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline
        self.visFr = visFr
        #self.clump_gap_tolerance = 1 # the number of frames that can be skipped for a clump to still be considered a single clump
        #self.clump_radius_scale = 2.0 # the factor with which to multiply error_x by to determine a radius in which points belong to the same clump
        #self.clump_radius_offset = 150. # an offset in nm to add to the the clump detection radius (useful for detection before shift correction)



        visFr.AddMenuItem('Extras', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')



    def OnClumpDBSCAN(self, event=None):
        eps_dlg = wx.NumberEntryDialog(None, 'aa', 'eps', 'cc', 1, 0, 9e9)
        eps_dlg.ShowModal()
        min_points_dlg = wx.NumberEntryDialog(None, 'aa', 'min_points', 'cc', 1, 0, 9e9)
        min_points_dlg.ShowModal()

        # get probe (color channel) to be clumped
        probe_dlg = wx.NumberEntryDialog(None, 'aa', 'probe channel to clump', 'cc', 1, 0, np.max(self.pipeline['probe']) + 1)

        probe_dlg.ShowModal()
        probeID = probe_dlg.GetValue()

        # generate boolean array of localizations to be searched
        if probeID is None:
            fi = self.pipeline.filter.Index
        else:
            fi = np.logical_and(self.pipeline.filter.Index, self.pipeline.selectedDataSource['probe'] == probeID)

        pyDeClump.dbscanClump(self.pipeline.selectedDataSource, float(eps_dlg.GetValue()),
                              float(min_points_dlg.GetValue()), filterIndex=fi)




def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.multiview = clumper(visFr)
