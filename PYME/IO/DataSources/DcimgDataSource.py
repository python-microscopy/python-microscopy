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

from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
from .BaseDataSource import XYTCDataSource

from PYME.IO import dcimg

class DataSource(XYTCDataSource):
    moduleName = 'DcimgDataSource'
    def __init__(self, filename, taskQueue=None):
        self.filename = getFullExistingFilename(filename)#convert relative path to full path
        
        self.DCIFile = dcimg.DCIMGFile(self.filename)
        
    def getSlice(self, ind):
        #print ind
        return self.DCIFile.get_frame(ind)

    def getSliceShape(self):
        return self.DCIFile.get_slice_shape()

    def getNumSlices(self):
        return self.DCIFile.get_num_slices()

    def getEvents(self):
        return []

    def release(self):
        del self.DCIFile

    def reloadData(self):
        self.release()
        self.DCIFile = dcimg.DCIMGFile(self.filename)
