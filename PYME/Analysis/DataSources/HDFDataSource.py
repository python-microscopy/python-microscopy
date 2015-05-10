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

from PYME.ParallelTasks.relativeFiles import getFullFilename
import tables
from .BaseDataSource import BaseDataSource

class DataSource(BaseDataSource):
    moduleName = 'HDFDataSource'
    def __init__(self, h5Filename, taskQueue=None):
        self.h5Filename = getFullFilename(h5Filename)#convert relative path to full path
        self.h5File = tables.openFile(self.h5Filename)

    def getSlice(self, ind):
        if ind >= self.h5File.root.ImageData.shape[0]:
                self.reloadData() #try reloading the data in case it's grown
        
        return self.h5File.root.ImageData[ind, :,:]


    def getSliceShape(self):
        return self.h5File.root.ImageData.shape[1:3]

    def getNumSlices(self):
        return self.h5File.root.ImageData.shape[0]

    def getEvents(self):
        return self.h5File.root.Events[:]

    def release(self):
        self.h5File.close()

    def reloadData(self):
        self.h5File.close()
        self.h5File = tables.openFile(self.h5Filename)
