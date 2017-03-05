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

class ParticleTracker:
    def __init__(self, visFr):
        self.visFr = visFr

        visFr.AddMenuItem('Extras', "Get segmented IDs from image", self.OnGetIDs)
        visFr.AddMenuItem('Extras', "Measure objects", self.OnMeasure)


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
        from PYME.IO import image
        from PYME.Analysis.points import objectMeasurements

        visFr = self.visFr
        pipeline = visFr.pipeline

        dlg = wx.SingleChoiceDialog(
                None, 'choose the image which contains labels', 'Use Segmentation',
                image.openImages.keys(),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            img = image.openImages[dlg.GetStringSelection()]

            ids, numPerObject = objectMeasurements.getIDs(pipeline, img)

            pipeline.addColumn('objectID', ids)
            pipeline.addColumn('NEvents', numPerObject[ids-1])

            pipeline.Rebuild()

        dlg.Destroy()

    def OnMeasure(self, event):
        from PYME.LMVis import objectMeasure

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





def Plug(visFr):
    """Plugs this module into the gui"""
    ParticleTracker(visFr)
