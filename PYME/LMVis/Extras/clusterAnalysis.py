
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


class ClusterAnalyser:
    """

    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline
        self.nearestNeighbourDistances = {}

        visFr.AddMenuItem('Extras', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')
        visFr.AddMenuItem('Extras', 'Nearest Neighbor Distances: two-species', self.OnNearestNeighborTwoSpecies,
                          helpText='')

        self.GeneratedMeasures = dict()

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

        chans = self.pipeline.colourFilter.getColourChans()
        nchan = len(chans)
        if nchan < 2:
            raise RuntimeError('NearestNeighborTwoSpecies requires two color channels')

        # select with GUI, as this allows flexibility of choosing which channel neighbor distances are with respect to
        chan_dlg = wx.TextEntryDialog(None, 'Pick two color channels for nearest neighbors distance calculation',
                                      'Nearest neighbor of colors[1] in colors[0]', ','.join(chans))
        chan_dlg.ShowModal()
        selectedChans = str(chan_dlg.GetValue()).replace(' ', '').split(',')
        if (selectedChans[0] not in chans) or (selectedChans[1] not in chans):
            raise RuntimeError('NearestNeighborTwoSpecies requires two color channels')

        dispColor = self.pipeline.colourFilter.currentColour
        
        self.pipeline.colourFilter.setColour(selectedChans[0])
        chan0 = {'x': self.pipeline['x'], 'y': self.pipeline['y'], 'z': self.pipeline['z']}
        
        self.pipeline.colourFilter.setColour(selectedChans[1])
        chan1 = {'x': self.pipeline['x'], 'y': self.pipeline['y'], 'z': self.pipeline['z']}
        
        namespace = {'A': chan0, 'B': chan1}

        # restore original display settings
        self.pipeline.colourFilter.setColour(dispColor)


        matchMaker = measurement.NearestNeighbourDistances(columns=['x', 'y', 'z'], inputChan0='A', inputChan1='B', outputname='output',key='neighbourDists')
        matchMaker.execute(namespace)

        self.nearestNeighbourDistances[selectedChans] = np.array(namespace['output']['neighbourDists'])

        print 'Results are stored in clusterAnalyser.nearestNeighbourDistances[%s]' % selectedChans


def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.clusterAnalyser = ClusterAnalyser(visFr)

