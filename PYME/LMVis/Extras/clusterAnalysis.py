
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
import sys


class ClusterAnalyser:
    """

    """
    def __init__(self, visFr):
        self.visFr = visFr
        self.pipeline = visFr.pipeline
        self.nearestNeighbourDistances = {}
        self.colocalizationRatios = {}

        visFr.AddMenuItem('Extras>DBSCAN', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')
        visFr.AddMenuItem('Extras>DBSCAN', 'Nearest Neighbor Distances- two-species', self.OnNearestNeighborTwoSpecies,
                          helpText='')
        visFr.AddMenuItem('Extras>DBSCAN', 'DBSCAN - find mixed clusters', self.OnFindMixedClusters,
                          helpText='')
        visFr.AddMenuItem('Extras>DBSCAN', 'DBSCAN - clumps in time', self.OnClustersInTime,
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
            
        selectedChans = [chans[ci] for ci in chan_dlg.GetSelections()]
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


        matchMaker = measurement.NearestNeighbourDistances(columns=['x', 'y', 'z'], inputChan0='A', inputChan1='B',
                                                           outputName='output',key='neighbourDists')
        matchMaker.execute(namespace)

        self.nearestNeighbourDistances[''.join(selectedChans)] = np.array(namespace['output']['neighbourDists'])

        print 'Results are stored in clusterAnalyser.nearestNeighbourDistances[%s]' % selectedChans

    def OnFindMixedClusters(self, event=None):
        """
        FindMixedClusters first uses DBSCAN clustering on two color channels separately for denoising purposes, then
        after having removed noisy points, DBSCAN is run again on both channels combined, and the fraction of clumps
        containing both colors is determined.
        """
        from PYME.recipes import tablefilters
        from PYME.recipes.base import ModuleCollection
        import wx

        chans = self.pipeline.colourFilter.getColourChans()
        nchan = len(chans)
        if nchan < 2:
            raise RuntimeError('FindMixedClusters requires at least two color channels')
        else:
            selectedChans = [0, 1]


        #rad_dlg = wx.NumberEntryDialog(None, 'Search Radius For Core Points', 'rad [nm]', 'rad [nm]', 125, 0, 9e9)
        #rad_dlg.ShowModal()
        searchRadius = 125.0 #rad_dlg.GetValue()
        #minPt_dlg = wx.NumberEntryDialog(None, 'Minimum Points To Be Core Point', 'min pts', 'min pts', 3, 0, 9e9)
        #minPt_dlg.ShowModal()
        minClumpSize = 3 #minPt_dlg.GetValue()

        #build a recipe programatically
        rec = ModuleCollection()
        #split input according to colour channels
        rec.add_module(tablefilters.ExtractTableChannel(rec, inputName='input', outputName='chan0', channel=chans[0]))
        rec.add_module(tablefilters.ExtractTableChannel(rec,inputName='input', outputName='chan1', channel=chans[1]))

        #clump each channel
        rec.add_module(tablefilters.DBSCANClustering(rec,inputName='chan0', outputName='chan0_clumped',
                                                         searchRadius=searchRadius, minClumpSize=minClumpSize))
        rec.add_module(tablefilters.DBSCANClustering(rec,inputName='chan1', outputName='chan1_clumped',
                                                         searchRadius=searchRadius, minClumpSize=minClumpSize))

        #filter unclumped points
        rec.add_module(tablefilters.FilterTable(rec,inputName='chan0_clumped', outputName='chan0_cleaned',
                                                         filters={'dbscanClumpID' : [.5, sys.maxint]}))
        rec.add_module(tablefilters.FilterTable(rec,inputName='chan1_clumped', outputName='chan1_cleaned',
                                               filters={'dbscanClumpID': [.5, sys.maxint]}))

        #rejoin cleaned datasets
        rec.add_module(tablefilters.ConcatenateTables(rec,inputName0='chan0_cleaned', inputName1='chan1_cleaned',
                                                      outputName='joined'))

        #clump on cleaded and rejoined data
        rec.add_module(tablefilters.DBSCANClustering(rec,inputName='joined', outputName='output',
                                                     searchRadius=searchRadius, minClumpSize=minClumpSize))


        rec.namespace['input'] = self.pipeline #do it before configuring so that we already have the channe; names populated
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return #handle cancel

        #run recipe
        joined_clumps = rec.execute()

        joined_clump_IDs = np.unique(joined_clumps['dbscanClumpID'])
        joined_clump_IDs = joined_clump_IDs[joined_clump_IDs > .5] #reject unclumped points

        chan0_clump_IDs = np.unique(joined_clumps['dbscanClumpID'][joined_clumps['concatSource'] < .5])
        chan0_clump_IDs = chan0_clump_IDs[chan0_clump_IDs > .5]

        chan1_clump_IDs = np.unique(joined_clumps['dbscanClumpID'][joined_clumps['concatSource'] > .5])
        chan1_clump_IDs = chan1_clump_IDs[chan1_clump_IDs > .5]

        both_chans_IDS = [c for c in chan0_clump_IDs if c in chan1_clump_IDs]

        n_total_clumps = len(joined_clump_IDs)

        print('Total clumps: %i' % n_total_clumps)
        c0Ratio = float(len(chan0_clump_IDs)) / n_total_clumps
        print('fraction clumps with channel %i present: %f' % (selectedChans[0], c0Ratio))
        self.colocalizationRatios['Channel%iin%i%i' % (selectedChans[0], selectedChans[0], selectedChans[1])] = c0Ratio

        c1Ratio = float(len(chan1_clump_IDs)) / n_total_clumps
        print('fraction clumps with channel %i present: %f' % (selectedChans[1], c1Ratio))
        self.colocalizationRatios['Channel%iin%i%i' % (selectedChans[1], selectedChans[0], selectedChans[1])] = c1Ratio

        bothChanRatio = float(len(both_chans_IDS)) / n_total_clumps
        print( 'fraction of clumps with both channel %i and %i present: %f' % (selectedChans[0], selectedChans[1], bothChanRatio))
        self.colocalizationRatios['mixedClumps%i%i' % tuple(selectedChans)] = bothChanRatio

        self._rec = rec

    def OnClustersInTime(self, event=None):
        from PYME.recipes import tablefilters, measurement
        from PYME.recipes.base import ModuleCollection
        import matplotlib.pyplot as plt

        # build a recipe programatically
        rec = ModuleCollection()

        # split input according to colour channel selected
        rec.add_module(tablefilters.ExtractTableChannel(rec, inputName='input', outputName='chan0',
                                                              channel='chan0'))

        rec.add_module(measurement.ClumpsInTime(rec, inputName='chan0', stepSize=3000, searchRadius=75,
                                                minPtsPerClump=3, outputName='output'))


        rec.namespace['input'] = self.pipeline #do before configuring so that we already have the channel names populated
        #configure parameters
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return #handle cancel

        incrementedClumps = rec.execute()

        plt.figure()
        plt.scatter(incrementedClumps['t'], incrementedClumps['N_origClustersWithMinPoints'],
                    label='Original Clusters with N > minPts', edgecolors='r', facecolors='none', marker='s')
        plt.scatter(incrementedClumps['t'], incrementedClumps['N_rawDBSCAN'], label='raw DBSCAN', edgecolors='b',
                    facecolors='b', marker='x')
        plt.scatter(incrementedClumps['t'], incrementedClumps['N_origClusterDBSCAN'],
                    label='DBSCAN on points originally clustered', edgecolors='g', facecolors='none')
        plt.legend(loc=4, scatterpoints=1)
        plt.xlabel('Number of frames included')
        plt.ylabel('Number of Clusters')
        plt.title('minPoints=%i, searchRadius = %.0f nm' % (rec.modules[-1].minPtsPerClump, rec.modules[-1].searchRadius))



def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.clusterAnalyser = ClusterAnalyser(visFr)

