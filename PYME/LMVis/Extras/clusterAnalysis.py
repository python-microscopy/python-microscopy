
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

import numpy as np


class ClusterAnalyser:
    """

    """
    def __init__(self, visFr):
        self.visFr = visFr
        self.pipeline = visFr.pipeline
        self.nearestNeighbourDistances = {}

        visFr.AddMenuItem('Extras', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')
        visFr.AddMenuItem('Extras', 'Nearest Neighbor Distances- two-species', self.OnNearestNeighborTwoSpecies,
                          helpText='')
        visFr.AddMenuItem('Extras', 'DBSCAN - find mixed clusters', self.OnFindMixedClusters,
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
        chan_dlg = wx.MultiChoiceDialog(self.visFr, 'Pick two color channels for nearest neighbors distance calculation',
                                      'Nearest neighbour channel selection', chans)
        chan_dlg.SetSelections([0,1])
        if not chan_dlg.ShowModal() == wx.ID_OK:
            return #need to handle cancel
            
        selectedChans = chans[chan_dlg.GetSelections()]
        if not len(selectedChans) == 2:
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

    def OnFindMixedClusters(self, event=None):
        """

        """
        from PYME.recipes import tablefilters
        import wx
        import numpy as np

        clumper = tablefilters.DBSCANClustering()

        namespace = {'pipeline': self.pipeline}
        clumper.inputName = 'pipeline'

        print 'Input DBSCAN parameters for channel 0'
        # ignore other channels for now
        self.pipeline.filterKeys['probe'] = (-0.5, 0.5)
        self.pipeline.Rebuild()

        rad_dlg = wx.NumberEntryDialog(None, 'Search Radius For Core Points', 'rad [nm]', 'rad [nm]', 125, 0, 9e9)
        rad_dlg.ShowModal()
        clumper.searchRadius = rad_dlg.GetValue()
        minPt_dlg = wx.NumberEntryDialog(None, 'Minimum Points To Be Core Point', 'min pts', 'min pts', 3, 0, 9e9)
        minPt_dlg.ShowModal()
        clumper.minPtsForCore = minPt_dlg.GetValue()
        clumper.outputName = 'chan0Clumps'
        clumper.execute(namespace)
        self.pipeline.addColumn(clumper.outputName, namespace[clumper.outputName]['dbscanClumpID'])
        # filter unclumped points from this channel
        self.pipeline.filterKeys[clumper.outputName] = (-0.5, np.max(self.pipeline[clumper.outputName]) + 1)
        self.pipeline.Rebuild()

        print 'Input DBSCAN parameters for channel 1'
        # ignore other channels for now
        self.pipeline.filterKeys['probe'] = (0.5, 1.5)
        self.pipeline.Rebuild()
        rad_dlg.ShowModal()
        minPt_dlg.ShowModal()
        clumper.searchRadius = rad_dlg.GetValue()
        clumper.minPtsForCore = minPt_dlg.GetValue()
        clumper.outputName = 'chan1Clumps'
        clumper.execute(namespace)
        self.pipeline.addColumn(clumper.outputName, namespace[clumper.outputName]['dbscanClumpID'])
        # filter unclumped points from this channel
        self.pipeline.filterKeys[clumper.outputName] = (-0.5, np.max(self.pipeline[clumper.outputName]) + 1)


        # Clump both colors together

        # add back in all colors
        del self.pipeline.filterKeys['probe']
        self.pipeline.Rebuild()
        print 'Input DBSCAN parameters for combined channel clumping'
        rad_dlg.ShowModal()
        minPt_dlg.ShowModal()
        clumper.searchRadius = rad_dlg.GetValue()
        clumper.minPtsForCore = minPt_dlg.GetValue()
        clumper.outputName = 'mixedClumps'
        clumper.execute(namespace)
        self.pipeline.addColumn(clumper.outputName, namespace[clumper.outputName]['dbscanClumpID'])
        # filter noisy clumps (shouldn't be any to begin with if appropriate value of minPtsForcore is used
        self.pipeline.filterKeys[clumper.outputName] = (0, np.max(self.pipeline[clumper.outputName]) + 1)

        totMixed = np.unique(self.pipeline[clumper.outputName])
        self.pipeline.colourFilter.setColour('chan0')
        c0Clumps = np.unique(self.pipeline[clumper.outputName])
        self.pipeline.colourFilter.setColour('chan1')
        c1Clumps = np.unique(self.pipeline[clumper.outputName])
        print('Total clumps: %i' % len(totMixed))

        c0Ratio = float(len([c for c in c0Clumps if c in totMixed]))/len(totMixed)
        print c0Ratio

        c1Ratio = float(len([c for c in c1Clumps if c in totMixed]))/len(totMixed)
        print c1Ratio

        bothChanRatio = float(len([c for c in totMixed if ((c in c0Clumps) and (c in c1Clumps))]))/len(totMixed)
        print bothChanRatio

        self.pipeline.colourFilter.setColour('Everything')


def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.clusterAnalyser = ClusterAnalyser(visFr)

