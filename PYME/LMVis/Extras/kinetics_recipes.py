from PYME.recipes.base import register_module, ModuleBase, Filter, OutputModule
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr, Button, ToolbarButton
import matplotlib as plt
import wx
import numpy as np
import pylab
from PIL import Image
from PYME.IO import tabular

from PYME.recipes.tablefilters import FilterTable
from PYME.recipes.base import ModuleCollection

from PYME.recipes import kinetics_recipes as kin_rcps
from PYME.recipes.localisations import DBSCANClustering as dbscan
from PYME.recipes.output import HDFOutput

class DyeKinetics(object):
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Dye Kinetics', 'Find on and off time durations for molecules', self.dye_kinetics, helpText='')

    def dye_kinetics(self, event):
        import sys
        from time import time

        t0 = time()

        self.pipeline = self.visFr.pipeline

        recipe = self.visFr.pipeline.recipe

        script_init = kin_rcps.DyeKineticsInit(recipe, inputName=self.pipeline.selectedDataSourceKey, outputName='localisations',
                                             clusteringRadius=40.0, blinkRadius=20.0, clusterColumnName='clusterID', fitHistograms=True,
                                             blinkGapTolerance=0, onTimesColumnName='on_times', offTimesColumnName='off_times',
                                             minimumClusterSize=2, minimumKeptBlinkSize=2, minimumOffTimeInSecondsFit=0,
                                             maximumOffTimeInSecondsToFit=10000.0, offTimeFitBinWidthInSeconds=1)

        if script_init.configure_traits(kind='modal'):

            recipe.trait_set(execute_on_invalidation=False)

            recipe.add_module(script_init)

            recipe.add_module(kin_rcps.FindClumps_DK(recipe, inputName='localisations', outputName='with_clumps',
                                                   timeWindow=script_init.blinkGapTolerance, clumpRadiusVariable=u'1.0',
                                                   clumpRadiusScale=script_init.blinkRadius, minClumpSize=script_init.minimumKeptBlinkSize))

            minblink = script_init.minimumKeptBlinkSize

            minblink -= 1

            recipe.add_module(FilterTable(recipe, inputName='with_clumps', outputName='good_clumps',
                                                    filters={'clumpSize': [minblink, sys.maxsize]}))


            recipe.add_module(kin_rcps.MergeClumpsDyeKinetics(recipe, inputName='good_clumps', outputName='coalesced', lebelKey='clumpIndex'))

            recipe.add_module(FilterTable(recipe, inputName='coalesced', outputName='filtered_coalesced',
                                          filters={'clumpSize': [minblink, sys.maxsize]}))

            try:
                import hdbscan
            except:
                print('HDBScan not installed, defaulting to regular DBScan')
                clusterer = dbscan(recipe, inputName='filtered_coalesced', searchRadius=script_init.clusteringRadius,
                                   minClumpSize=script_init.minimumClusterSize,
                                   clumpColumnName=script_init.clusterColumnName, outputName='clusters')

            else:
                print('HDBScan installed, clustering with that')
                clusterer = kin_rcps.HDBSCANClustering(recipe, input_name='filtered_coalesced',
                                                     search_radius=script_init.clusteringRadius,
                                                     min_clump_size=script_init.minimumClusterSize,
                                                     clump_column_name=script_init.clusterColumnName,
                                                     output_name='clusters')

            recipe.add_module(clusterer)

            recipe.add_module(FilterTable(recipe, inputName='clusters', outputName='filt_clusters',
                                                    filters={script_init.clusterColumnName: [0.0, sys.maxsize]}))

            recipe.add_module(kin_rcps.FindBlinkStateDurations(recipe, inputName='filt_clusters', outputName='unfiltered_kinetics',
                                                             labelKey=script_init.clusterColumnName, onTimesColName=script_init.onTimesColumnName,
                                                             offTimesColName=script_init.offTimesColumnName))

            recipe.add_module(FilterTable(recipe, inputName='unfiltered_kinetics', outputName='filtered_kinetics',
                                          filters={script_init.offTimesColumnName: [0.0, sys.maxsize],
                                                   script_init.onTimesColumnName: [minblink, sys.maxsize]}))

            recipe.add_module(kin_rcps.FitSwitchingRates(recipe, inputName='filtered_kinetics', outputName='kinetics_out',
                                                       onTimesColName=script_init.onTimesColumnName,
                                                       offTimesColName=script_init.offTimesColumnName,
                                                       minOffTimeSecondsToFit=script_init.minimumOffTimeInSecondsToFit,
                                                       maxOffTimeSecondsToFit=script_init.maximumOffTimeInSecondsToFit,
                                                       offTimesFitBinSizeInSeconds=script_init.offTimeFitBinWidthInSeconds))
            recipe.execute()
            self.visFr.pipeline.selectDataSource('filtered_kinetics')

        print 'RUNTIME OF ' + str(time()-t0)

class ExportEmphist(object):
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Dye Kinetics', 'Save empirical histogram JSON file to disk', self.exp_emphist, helpText='')

    def exp_emphist(self, event):

        self.pipeline = self.visFr.pipeline

        recipe = self.visFr.pipeline.recipe

        json_exporter = kin_rcps.save_EmpHistJson(recipe, inputName='emphist_data', outputName='emphist_file', outputdir='C:\\Users\\bdr25\\Desktop\\data\\JSON files')

        if json_exporter.configure_traits(kind='modal'):

            recipe.trait_set(execute_on_invalidation=False)

            recipe.add_module(json_exporter)

            recipe.execute()


def Plug(visFr):
    visFr.dye_kinetics_manger = DyeKinetics(visFr)
    visFr.exp_emphist_manager = ExportEmphist(visFr)
