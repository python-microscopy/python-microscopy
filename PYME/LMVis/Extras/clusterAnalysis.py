
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
        self.nearestNeighbourDistances = []
        self.colocalizationRatios = {}
        self.pairwiseDistances = {}
        self.clusterMeasures = []

        visFr.AddMenuItem('Analysis>Clustering', 'DBSCAN Clump', self.OnClumpDBSCAN,
                          helpText='')
        visFr.AddMenuItem('Analysis>Clustering', 'DBSCAN - find mixed clusters', self.OnFindMixedClusters,
                          helpText='')
        visFr.AddMenuItem('Analysis>Clustering', 'Cluster count vs. imaging time', self.OnClustersInTime,
                          helpText='')
        visFr.AddMenuItem('Analysis>Clustering', 'Pairwise Distance Histogram', self.OnPairwiseDistanceHistogram,
                          helpText='')
        visFr.AddMenuItem('Analysis>Clustering', 'Nearest Neighbor Distance Histogram', self.OnNearestNeighbor,
                          helpText='')
        visFr.AddMenuItem('Analysis>Clustering', 'Measure Clusters', self.OnMeasureClusters,
                          helpText='')


    def OnClumpDBSCAN(self, event=None):
        """
        Runs sklearn DBSCAN clustering algorithm on pipeline filtered results using the GUI defined in the DBSCAN
        recipe module.

        Args are user defined through GUI
            eps: search radius for clustering
            min_points: number of points within eps required for a given point to be considered a core point

        """
        from PYME.recipes import localisations

        clumper = localisations.DBSCANClustering()
        if clumper.configure_traits(kind='modal'):
            namespace = {clumper.inputName: self.pipeline}
            clumper.execute(namespace)

            self.pipeline.addColumn(clumper.outputName, namespace[clumper.outputName]['dbscanClumpID'])

    def OnNearestNeighbor(self, event=None):
        """
        GUI front-end for the NearestNeighbourDistances recipe module. Handling is in place for single species nearest
        neighbour calculations, as well as two species where the user is queried via the GUI as to which channels to use
        for building and querying the kdtree.
        """
        from PYME.recipes import measurement
        import matplotlib.pyplot as plt
        import wx

        dkey = 'neighbourDists'

        chans = self.pipeline.colourFilter.getColourChans()
        selectedChans = chans
        if len(chans) >= 2:
            # select with GUI, as this allows flexibility of choosing which channel neighbor distances are with respect to
            chan_dlg = wx.MultiChoiceDialog(self.visFr, 'Pick 2 channels for two species, or select none to ignore colour.',
                                      'Nearest neighbour channel selection', chans)
            chan_dlg.SetSelections([0,1])
            if not chan_dlg.ShowModal() == wx.ID_OK:
                return  # handle cancel

            selectedChans = chan_dlg.GetSelections()
            if len(selectedChans) == 2:
                # select order
                order_dlg = wx.SingleChoiceDialog(self.visFr, 'Choose which perspective to measure from',
                                                  'Nearest neighbour perspective selection',
                                                  ['Nearest %s to each %s' % (selectedChans[0], selectedChans[1]),
                                                   'Nearest %s to each %s' % (selectedChans[1], selectedChans[0])])
                if not order_dlg.ShowModal() == wx.ID_OK:
                    return  # handle cancel
                order = order_dlg.GetSelection()
            
                selectedChans = [chans[ci] for ci in selectedChans]
                if order:
                    selectedChans.reverse()

                desc = 'Nearest %s to each %s' % (selectedChans[0], selectedChans[1])

                dispColor = self.pipeline.colourFilter.currentColour

                self.pipeline.colourFilter.setColour(selectedChans[0])
                chan0 = {'x': self.pipeline['x'], 'y': self.pipeline['y'], 'z': self.pipeline['z']}

                self.pipeline.colourFilter.setColour(selectedChans[1])
                chan1 = {'x': self.pipeline['x'], 'y': self.pipeline['y'], 'z': self.pipeline['z']}

                namespace = {'A': chan0, 'B': chan1}

                # restore original display settings
                self.pipeline.colourFilter.setColour(dispColor)

                matchMaker = measurement.NearestNeighbourDistances(columns=['x', 'y', 'z'], inputChan0='A', inputChan1='B',
                                                               outputName='output', key=dkey)
            elif len(selectedChans) > 2:
                raise RuntimeError('NearestNeighbour cannot handle more than two channels')


        if len(selectedChans) < 2:
            desc = 'Single channel nearest neighbours'
            namespace = {'A': self.pipeline}
            matchMaker = measurement.NearestNeighbourDistances(columns=['x', 'y', 'z'], inputChan0='A', inputChan1='',
                                                               outputName='output', key=dkey)


        matchMaker.execute(namespace)
        dists = np.array(namespace['output'][dkey])

        self.nearestNeighbourDistances.append({'neighbourDists': dists, 'description': desc})

        print('Results are stored in clusterAnalyser.nearestNeighbourDistances[%i]' % (len(self.nearestNeighbourDistances) - 1))
        plt.hist(dists, range=[0, np.mean(dists) + 3.5*np.std(dists)])

    def OnFindMixedClusters(self, event=None):
        """
        FindMixedClusters first uses DBSCAN clustering on two color channels separately for denoising purposes, then
        after having removed noisy points, DBSCAN is run again on both channels combined, and the fraction of clumps
        containing both colors is determined.
        """
        from PYME.recipes import tablefilters, localisations
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
        rec.add_module(localisations.ExtractTableChannel(rec, inputName='input', outputName='chan0', channel=chans[0]))
        rec.add_module(localisations.ExtractTableChannel(rec,inputName='input', outputName='chan1', channel=chans[1]))

        #clump each channel
        rec.add_module(localisations.DBSCANClustering(rec,inputName='chan0', outputName='chan0_clumped',
                                                         searchRadius=searchRadius, minClumpSize=minClumpSize))
        rec.add_module(localisations.DBSCANClustering(rec,inputName='chan1', outputName='chan1_clumped',
                                                         searchRadius=searchRadius, minClumpSize=minClumpSize))

        #filter unclumped points
        rec.add_module(tablefilters.FilterTable(rec,inputName='chan0_clumped', outputName='chan0_cleaned',
                                                         filters={'dbscanClumpID' : [.5, sys.maxsize]}))
        rec.add_module(tablefilters.FilterTable(rec,inputName='chan1_clumped', outputName='chan1_cleaned',
                                               filters={'dbscanClumpID': [.5, sys.maxsize]}))

        #rejoin cleaned datasets
        rec.add_module(tablefilters.ConcatenateTables(rec,inputName0='chan0_cleaned', inputName1='chan1_cleaned',
                                                      outputName='joined'))

        #clump on cleaded and rejoined data
        rec.add_module(localisations.DBSCANClustering(rec,inputName='joined', outputName='output',
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

    def OnPairwiseDistanceHistogram(self, event=None):
        from PYME.recipes import tablefilters, localisations, measurement
        from PYME.recipes.base import ModuleCollection
        import matplotlib.pyplot as plt

        # build a recipe programatically
        distogram = ModuleCollection()

        # split input according to colour channels selected
        distogram.add_module(localisations.ExtractTableChannel(distogram, inputName='input', outputName='chan0',
                                                              channel='chan0'))
        distogram.add_module(localisations.ExtractTableChannel(distogram, inputName='input', outputName='chan1',
                                                              channel='chan0'))

        # Histogram
        distogram.add_module(measurement.PairwiseDistanceHistogram(distogram, inputPositions='chan0',
                                                                   inputPositions2='chan1', outputName='output'))

        distogram.namespace['input'] = self.pipeline #do before configuring so that we already have the channel names populated
        #configure parameters
        if not distogram.configure_traits(view=distogram.pipeline_view, kind='modal'):
            return #handle cancel
        selectedChans = (distogram.modules[-1].inputPositions, distogram.modules[-1].inputPositions2)
        #run recipe
        distances = distogram.execute()

        binsz = (distances['bins'][1] - distances['bins'][0])
        self.pairwiseDistances[selectedChans] = {'counts': np.array(distances['counts']),
                                                        'bins': np.array(distances['bins'] + 0.5*binsz)}

        plt.figure()
        plt.bar(self.pairwiseDistances[selectedChans]['bins'] - 0.5*binsz,
                self.pairwiseDistances[selectedChans]['counts'], width=binsz)



    def OnClustersInTime(self, event=None):
        #FIXME - this would probably be better in an addon module outside of the core project
        from PYME.recipes import localisations
        from PYME.recipes.base import ModuleCollection
        import matplotlib.pyplot as plt

        # build a recipe programatically
        rec = ModuleCollection()

        # split input according to colour channel selected
        rec.add_module(localisations.ExtractTableChannel(rec, inputName='input', outputName='chan0',
                                                              channel='chan0'))

        rec.add_module(localisations.ClusterCountVsImagingTime(rec, inputName='chan0', stepSize=3000, outputName='output'))


        rec.namespace['input'] = self.pipeline #do before configuring so that we already have the channel names populated
        #configure parameters
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return #handle cancel

        incrementedClumps = rec.execute()

        plt.figure()
        plt.scatter(incrementedClumps['t'], incrementedClumps['N_labelsWithLowMinPoints'],
                    label=('clusters with Npoints > %i' % rec.modules[-1].lowerMinPtsPerCluster), c='b', marker='s')
        plt.scatter(incrementedClumps['t'], incrementedClumps['N_labelsWithHighMinPoints'],
                    label=('clusters with Npoints > %i' % rec.modules[-1].higherMinPtsPerCluster), c='g', marker='o')

        plt.legend(loc=4, scatterpoints=1)
        plt.xlabel('Number of frames included')
        plt.ylabel('Number of Clusters')
        #plt.title('minPoints=%i, searchRadius = %.0f nm' % (, rec.modules[-1].higherMinPtsPerCluster))

    def OnMeasureClusters(self, event=None):
        """

        Calculates various measures for clusters using PYME.recipes.localisations.MeasureClusters

        Parameters
        ----------
        labelsKey: pipeline key to access array of label assignments. Measurements will be calculated for each label.


        """
        from PYME.recipes import localisations
        from PYME.recipes.base import ModuleCollection

        # build a recipe programatically
        measrec = ModuleCollection()

        measrec.add_module(localisations.MeasureClusters3D(measrec, inputName='input', labelsKey='dbscanClustered',
                                                       outputName='output'))

        measrec.namespace['input'] = self.pipeline
        #configure parameters
        if not measrec.configure_traits(view=measrec.pipeline_view, kind='modal'):
            return  # handle cancel

        # run recipe
        meas = measrec.execute()

        # For now, don't make this a data source, as that requires (for multicolor) clearing the pipeline mappings.
        self.clusterMeasures.append(meas)

        # plot COM points
        #self.visFr.glCanvas.setPoints3D(meas['x'], meas['y'], meas['z'], np.zeros_like(meas['x']))



def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.clusterAnalyser = ClusterAnalyser(visFr)

