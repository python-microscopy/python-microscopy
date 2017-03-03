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
        from PYME.IO import image

        visFr = self.visFr
        pipeline = visFr.pipeline

        dlg = wx.SingleChoiceDialog(
                None, 'choose the image which contains labels', 'Use Segmentation',
                image.openImages.keys(),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            img = image.openImages[dlg.GetStringSelection()]
            
            im_ox, im_oy, im_oz = img.origin
            
            #account for ROIs
            p_ox = pipeline.mdh['Camera.ROIPosX']*pipeline.mdh['voxelsize.x']*1e3
            p_oy = pipeline.mdh['Camera.ROIPosY']*pipeline.mdh['voxelsize.y']*1e3

            pixX = np.round((pipeline.mapping['x'] + p_ox - im_ox)/img.pixelSize).astype('i')
            pixY = np.round((pipeline.mapping['y'] + p_oy - im_oy)/img.pixelSize).astype('i')
            pixZ = np.round((pipeline.mapping['z'] - im_oz)/img.sliceSize).astype('i')
            
            if img.data.shape[2] == 1:
                #disregard z for 2D images
                pixZ = np.zeros_like(pixX)

            ind = (pixX < img.data.shape[0])*(pixY < img.data.shape[1])*(pixX >= 0)*(pixY >= 0)*(pixZ >= 0)*(pixZ < img.data.shape[2])

            ids = np.zeros_like(pixX)
            
            #assume there is only one channel
            ids[ind] = np.atleast_3d(img.data[:,:,:,0].squeeze())[pixX[ind], pixY[ind], pixZ[ind]].astype('i')

            numPerObject, b = np.histogram(ids, np.arange(ids.max() + 1.5) + .5)

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
