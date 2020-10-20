#!/usr/bin/python
##################
# objectMeasurements.py
#
# Copyright David Baddeley, 2011
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
import numpy as np

class ObjectMeasurer:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Analysis', "Get segmented IDs from image", self.OnGetIDs)
        visFr.AddMenuItem('Analysis', "Measure objects", self.OnMeasure)
        visFr.AddMenuItem('Analysis', 'Pairwise distance point features', self.gen_pairwise_distance_features)


    def OnGetIDs(self, event):
        """

        Function to propagate labels from a segmented image (or stack of images) to localizations within the pipeline.
        Localizations in the same area (or volume) of image.ImageStack labels will be given the same 'ObjectID' as that
        label. The ImageStack containing labels is selected through the GUI.

        Parameters
        ----------
        event: GUI event

        Returns
        -------
        Nothing, but adds ObjectID and NEvents columns to the pipeline
            ObjectID: Label number from image, mapped to each localization within that label
            NEvents: Number of localizations within the label that a given localization belongs to

        """
        from PYME.IO import image, tabular
        from PYME.recipes.localisations import LabelsFromImage
        #from PYME.Analysis.points import objectMeasure

        visFr = self.visFr
        pipeline = visFr.pipeline
        recipe = pipeline.recipe

        dlg = wx.SingleChoiceDialog(
                None, 'choose the image which contains labels', 'Use Segmentation',
                image.openImages.keys(),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            img_name = dlg.GetStringSelection()
            img = image.openImages[img_name]
            
            recipe.namespace[img_name] = img
            
            output_name = recipe.new_output_name('labels_from_img')
            mod = LabelsFromImage(recipe, inputName=pipeline.selectedDataSourceKey,inputImage=img_name, outputName=output_name)
            recipe.add_modules_and_execute([mod,])

            pipeline.selectDataSource(mod.outputName)

        dlg.Destroy()

    def OnMeasure(self, event):
        from PYME.Analysis.points import objectMeasure

        pipeline = self.visFr.pipeline

        chans = pipeline.colourFilter.getColourChans()

        ids = set(pipeline.mapping['objectID'].astype('i'))
        pipeline.objectMeasures = {}

        if len(chans) == 0:
            pipeline.objectMeasures['Everything'] = objectMeasure.measureObjectsByID(pipeline.colourFilter, 10,ids)
        else:
            curChan = pipeline.colourFilter.currentColour

            chanNames = chans[:]

#            if 'Sample.Labelling' in metadata.getEntryNames():
#                lab = metadata.getEntry('Sample.Labelling')
#
#                for i in range(len(lab)):
#                    if lab[i][0] in chanNames:
#                        chanNames[chanNames.index(lab[i][0])] = lab[i][1]

            for ch, i in zip(chans, range(len(chans))):
                pipeline.colourFilter.setColour(ch)
                #fitDecayChan(colourFilter, metadata, chanNames[i], i)
                pipeline.objectMeasures[chanNames[i]] = objectMeasure.measureObjectsByID(pipeline.colourFilter, 10,ids)
            
            pipeline.colourFilter.setColour(curChan)
            
    
    def gen_pairwise_distance_features(self, event=None):
        from PYME.recipes import machine_learning
        from PYME.ui import progress
        visFr = self.visFr
        pipeline = visFr.pipeline

        with progress.ComputationInProgress(visFr, 'calculating pairwise distance point features'):
            m = machine_learning.PointFeaturesPairwiseDist(pipeline.recipe, inputLocalisations=pipeline.selectedDataSourceKey, outputName=pipeline.new_ds_name('features'))
            m.edit_no_invalidate()
       
            pipeline.recipe.add_modules_and_execute([m,])
            
            pipeline.selectDataSource(m.outputName)
            
            if m.PCA:
                # we did PCA - display the principle component vectors
                import matplotlib.pyplot as plt
                
                plt.figure()
                plt.plot(m.binWidth * np.arange(m.numBins), pipeline.selectedDataSource.pca.components_.T)
                plt.legend(['pc%d' % i for i in range(pipeline.selectedDataSource.pca.components_.shape[0])])
                plt.grid()
                plt.xlabel('Distance [nm]')
                plt.ylabel('Excess density [a.u.]')
            
            





def Plug(visFr):
    """Plugs this module into the gui"""
    ObjectMeasurer(visFr)
