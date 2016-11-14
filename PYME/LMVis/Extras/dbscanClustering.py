
##################
# dbscanClustering.py
#
# Copyright David Baddeley, Andrew Barentine
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
from sklearn.cluster import dbscan


class DBSCANer:
    """
    Provides GUI handling for sklearn's DBSCAN clustering function
    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline

        visFr.AddMenuItem('Extras', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')

    def OnClumpDBSCAN(self, event=None):
        """
        Runs sklearn DBSCAN clustering algorithm pipeline x, y, and z output (post filter).

        Args are user defined through GUI
            eps: search radius for clustering
            min_points: number of points within eps required for a given point to be considered a core point

        """
        eps_dlg = wx.NumberEntryDialog(None, 'DBSCAN parameters', 'eps [nm]', 'esp [nm]', 1, 0, 9e9)
        eps_dlg.ShowModal()
        min_points_dlg = wx.NumberEntryDialog(None, 'DBSCAN parameters', 'min_points to be a core point', 'min_points', 1, 0, 9e9)
        min_points_dlg.ShowModal()

        # TODO: add dialog for choosing pipeline keys to cluster (and how many keys to cluster)

        # Note that sklearn gives unclustered points label of -1, and first value starts at 0.
        core_samp, dbLabels = dbscan(np.vstack([self.pipeline['x'], self.pipeline['y'], self.pipeline['z']]).T,
                                     eps_dlg.GetValue(), min_points_dlg.GetValue())

        # shift dbscan labels up by one, so that the 0th cluster doesn't get lost when pipeline fills in filtered points
        # need to move dbscan labels up by one, except for the noisy labels (-1), which we want to push down by one to
        # make room for pipeline giving zeros to currently filtered points that might be added back later (i.e. a color
        # channel)
        dbLabels[dbLabels == -1] = -2
        self.pipeline.addColumn('dbscanClumpID', dbLabels + 1)





def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.multiview = DBSCANer(visFr)

