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

#from PYME.IO.FileUtils.nameUtils import getFullFilename
from .BaseDataSource import BaseDataSource
#import tables
from PYME.IO.MetaDataHandler import get_camera_roi_origin

class DataSource(BaseDataSource):
    moduleName = 'FlatFieldDataSource'
    def __init__(self, parentSource, mdh, flatfield, dark = None):
        #self.h5Filename = getFullFilename(h5Filename)#convert relative path to full path
        #self.h5File = tables.openFile(self.h5Filename)
        self.source = parentSource
        self.mdh = mdh
        #self.flat = flatfield
        
        x0, y0 = get_camera_roi_origin(mdh)
        x1 = x0 + mdh.getOrDefault('Camera.ROIWidth', self.source.shape[0]) + 1
        y1 = y0 + mdh.getOrDefault('Camera.ROIHeight', self.source.shape[1]) + 1

        #print((x0, x1, y0, y1))

        #self.offset = mdh.getEntry()

        self.flat = flatfield[x0:x1, y0:y1]
        if dark is None:
            self.dark = self.mdh.getEntry('Camera.ADOffset')
        else:
            self.dark = dark[x0:x1, y0:y1]


    def getSlice(self, ind):
        #if ind >= self.h5File.root.ImageData.shape[0]:
        #        self.reloadData() #try reloading the data in case it's grown
        print((self.getSliceShape(), self.flat.shape))
        
        return (self.source.getSlice(ind) - self.dark)*self.flat


    def getSliceShape(self):
        return self.source.getSliceShape()

    def getNumSlices(self):
        return self.source.getNumSlices()

    def getEvents(self):
        return self.source.getEvents()

    def release(self):
        self.source.release()
