#!/usr/bin/python

##################
# HDFDataSource.py
#
# Copyright David Baddeley, 2009
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
""" A DataSource which crops multiple separated (but equal sized) ROIs out of a larger frame and then 
    recombines them. 

    Useful for situations such as multi-view imaging where there is blank space between 
    the views - e.g. the Bewersdorf lab biplane setup.

    Views are concatenated along the x-axis
"""

from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from .BaseDataSource import BaseDataSource
import numpy as np

# default cropping info - should correspond to the Bewersdorf lab setup
# IMPORTANT: this should be sourced from metadata, the code here serves only as an example

CROP_INFO_YU = {
    'Multiview.NumROIs' : 4,
    'Multiview.ROISize' : (256, 256),
    'Multiview.ROI0Origin' : (132, 0),
    'Multiview.ROI1Origin' : (637, 0),
    'Multiview.ROI2Origin' : (1087, 0),
    'Multiview.ROI3Origin' : (1672, 0),
}


class DataSource(BaseDataSource):
    moduleName = 'MultiviewDataSource'
    def __init__(self, dataSource, mdh=CROP_INFO_YU):
        self.ds = dataSource

        self.numROIs = mdh['Multiview.NumROIs']
        self.roiSizeX, self.roiSizeY = mdh['Multiview.ROISize']

        #set vertical ROI size to the minimum of the specified vertical ROI size and the
        #vertical size of the raw data. This allows us to re-use roi settings even if 
        #we have set a smaller vertical ROI on the camera
        self.roiSizeY = min(self.roiSizeY, self.ds.getSliceShape()[1])

        self.viewOrigins = [mdh['Multiview.ROI%dOrigin' % i] for i in range(self.numROIs)]
        
    def getSlice(self, ind):            
        f = self.ds.getSlice(ind)

        #extract ROIs and concatenate
        rois = [f[ox:(ox + self.roiSizeX), oy:(oy + self.roiSizeY)] for ox, oy in self.viewOrigins]
        return np.concatenate(rois, axis = 0)

    def getSliceShape(self):
        return (self.roiSizeX*self.numROIs, self.roiSizeY)

    def getNumSlices(self):
        return self.ds.getNumSlices()

    def getEvents(self):
        return self.ds.getEvents()

    def release(self):
        self.ds.release()

    def reloadData(self):
        self.ds.reloadData()
