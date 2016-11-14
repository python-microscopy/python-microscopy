
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
        Runs sklearn DBSCAN clustering algorithm on pipeline filtered results using the GUI defined in the DBSCAN
        recipe module.

        Args are user defined through GUI
            eps: search radius for clustering
            min_points: number of points within eps required for a given point to be considered a core point

        """
        from PYME.recipes import tablefilters
        namespace = {'filtered': self.pipeline}
        clumper = tablefilters.DBSCANClustering()
        if clumper.configure_traits(kind='modal'):
            clumper.execute(namespace)

            self.pipeline.addColumn('dbscanClumpID', namespace['dbscanClustered']['dbscanClumpID'])



def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.multiview = DBSCANer(visFr)

