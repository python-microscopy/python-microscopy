
##################
# clusterAnalysis.py
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

def splitDataSource(pipeline, keyToSplit):
    """

    Args:
        pipeline: data source or dictionary-like object
        keyToSplit: key to generate separate datasource based on. Unique values of pipeline[keyToSplit] will be given
            their own data source

    Returns:
        dSources: a list where each element is a data source corresponding to a value of np.unique(pipeline[keyToSplit])

    """
    from PYME.recipes import tablefilters
    import numpy as np

    sieve = tablefilters.Filter()
    sieve.inputName = 'pipeline'
    namespace = {'pipeline': pipeline}

    dSources = []
    for ci in np.unique(pipeline[keyToSplit]).astype(int):
        sieve.filters = {keyToSplit: [ci-0.5, ci+0.5]}
        sieve.outputName = 'species%s' % ci

        sieve.execute(namespace)

        dSources.append(namespace[sieve.outputName])

    return dSources


class ClusterAnalyser:
    """
    Provides GUI handling for sklearn's DBSCAN clustering function
    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline

        visFr.AddMenuItem('Extras', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')
        visFr.AddMenuItem('Extras', 'Nearest Neighbor Distances: two-species', self.OnNearestNeighborTwoSpecies,
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

        clumper = tablefilters.DBSCANClustering()
        if clumper.configure_traits(kind='modal'):
            namespace = {clumper.inputName: self.pipeline}
            clumper.execute(namespace)

            self.pipeline.addColumn(clumper.outputName, namespace[clumper.outputName]['dbscanClumpID'])

    def OnNearestNeighborTwoSpecies(self, event=None):
        """
        GUI front-end for the NearestNeighborTwoSpecies recipe module. Since the module requires two separate datasource
        or dictionary-like inputs, the selectedDataSource is split based on unique values of a given pipeline key (e.g.
        probe). If there are more than two unique elements in pipeline[keyToSplit], the nearest neighbor calculations
        will be cycled through as 01, 12, 23, ...
        """
        from PYME.recipes import measurement
        import wx

        split_dlg = wx.TextEntryDialog(None, 'Key to split species based on', 'Split Key', 'probe')
        split_dlg.ShowModal()
        column_dlg = wx.TextEntryDialog(None, 'Keys to use in distance calculations', 'Distance keys', 'x,y,z')
        column_dlg.ShowModal()

        dSources = splitDataSource(self.pipeline, str(split_dlg.GetValue()))

        matchMaker = measurement.NearestNeighbourDistances()
        matchMaker.columns = str(column_dlg.GetValue()).replace(' ', '').split(',')

        for di in range(len(dSources)-1):
            matchMaker.inputChan0 = 'in%s' % di
            matchMaker.inputChan1 = 'in%s' % (di + 1)
            matchMaker.outputName = 'neighbourDists_%s%s' % (di, di + 1)
            matchMaker.key = matchMaker.outputName

            namespace = {'in%s' % di: dSources[di], 'in%s' % (di + 1): dSources[di + 1]}

            matchMaker.execute(namespace)

            self.GeneratedMeasures['NearestNeighborDistances_%s%s' % (di, di + 1)] = namespace[matchMaker.outputName][matchMaker.outputName].as_matrix()

        print 'Nearest neighbor results have been stored in pipeline.visFr.clusterAnalyser.GeneratedMeasures[NearestNeighborDistances_##]'


def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.ClusterAnalyser = ClusterAnalyser(visFr)

