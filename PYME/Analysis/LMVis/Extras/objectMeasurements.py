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

        ID_GET_IDS = wx.NewId()
        visFr.extras_menu.Append(ID_GET_IDS, "Get segmented IDs from image")
        visFr.Bind(wx.EVT_MENU, self.OnGetIDs, id=ID_GET_IDS)

        ID_MEASURE = wx.NewId()
        visFr.extras_menu.Append(ID_MEASURE, "Measure objects")
        visFr.Bind(wx.EVT_MENU, self.OnMeasure, id=ID_MEASURE)

    def OnGetIDs(self, event):
        from PYME.DSView import image

        visFr = self.visFr
        pipeline = visFr.pipeline

        dlg = wx.SingleChoiceDialog(
                None, 'choose the image which contains labels', 'Use Segmentation',
                image.openImages.keys(),
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            img = image.openImages[dlg.GetStringSelection()]
            
            #account for ROIs
            dRx = pipeline.mdh['Camera.ROIPosX']*pipeline.mdh['voxelsize.x']*1e3 - img.mdh['Camera.ROIPosX']*img.mdh['voxelsize.x']*1e3
            dRy = pipeline.mdh['Camera.ROIPosY']*pipeline.mdh['voxelsize.y']*1e3 - img.mdh['Camera.ROIPosY']*img.mdh['voxelsize.y']*1e3

            pixX = np.round((pipeline.mapping['x'] - img.imgBounds.x0 - dRx)/img.pixelSize).astype('i')
            pixY = np.round((pipeline.mapping['y'] - img.imgBounds.y0 - dRy)/img.pixelSize).astype('i')

            ind = (pixX < img.data.shape[0])*(pixY < img.data.shape[1])*(pixX >= 0)*(pixY >= 0)

            ids = np.zeros_like(pixX)
            #assume there is only one channel
            ids[ind] = img.data[:,:,:,0].squeeze()[pixX[ind], pixY[ind]].astype('i')

            numPerObject, b = np.histogram(ids, np.arange(ids.max() + 1.5) + .5)

            pipeline.selectedDataSource.objectIDs = np.zeros(len(pipeline.selectedDataSource['x']))
            pipeline.selectedDataSource.objectIDs[pipeline.filter.Index] = ids

            pipeline.selectedDataSource.numPerObject = np.zeros(len(pipeline.selectedDataSource['x']))
            pipeline.selectedDataSource.numPerObject[pipeline.filter.Index] = numPerObject[ids-1]

            pipeline.selectedDataSource.setMapping('objectID', 'objectIDs')
            pipeline.selectedDataSource.setMapping('NEvents', 'numPerObject')

            visFr.RegenFilter()
            visFr.CreateFoldPanel()

        dlg.Destroy()

    def OnMeasure(self, event):
        from PYME.Analysis.LMVis import objectMeasure

        chans = self.pipeline.colourFilter.getColourChans()

        ids = set(self.pipeline.mapping['objectID'].astype('i'))
        self.pipeline.objectMeasures = {}

        if len(chans) == 0:
            self.pipeline.objectMeasures['Everything'] = objectMeasure.measureObjectsByID(self.pipeline.colourFilter, 10,ids)
        else:
            curChan = self.pipeline.colourFilter.currentColour

            chanNames = chans[:]

#            if 'Sample.Labelling' in metadata.getEntryNames():
#                lab = metadata.getEntry('Sample.Labelling')
#
#                for i in range(len(lab)):
#                    if lab[i][0] in chanNames:
#                        chanNames[chanNames.index(lab[i][0])] = lab[i][1]

            for ch, i in zip(chans, range(len(chans))):
                self.pipeline.colourFilter.setColour(ch)
                #fitDecayChan(colourFilter, metadata, chanNames[i], i)
                self.pipeline.objectMeasures[chanNames[i]] = objectMeasure.measureObjectsByID(self.visFr.colourFilter, 10,ids)
            self.pipeline.colourFilter.setColour(curChan)





def Plug(visFr):
    '''Plugs this module into the gui'''
    ParticleTracker(visFr)