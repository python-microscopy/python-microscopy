#!/usr/bin/python

##################
# TiffDataSource.py
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
# try:
#     import javabridge
#     import bioformats
#     #javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
# except:
#     pass

# Try/pass removed to reveal loading issues
# unclear why try ... pass block was present in the first place TODO - check for potential regressions caused by removal
import javabridge
import bioformats

numVMRefs = 0

def ensure_VM():
    global numVMRefs
    
    if numVMRefs <1:
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    
    numVMRefs += 1
    
def release_VM():
    global numVMRefs
    numVMRefs -=1
    
    if numVMRefs < 1:
        javabridge.kill_vm()
    


from PYME.IO.FileUtils.nameUtils import getFullExistingFilename
#from PYME.IO.FileUtils import readTiff
#import Image
#from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon

import numpy as np

#from PYME.gohlke import tifffile


from .BaseDataSource import XYTCDataSource


class BioformatsFile(bioformats.ImageReader):
    def __init__(self, *args, **kwargs):
        # self.path is always the file path or the url, whichever is specified
        # url overrides a file path 
        ensure_VM()
        self._md = None
        self._series_count = None
        self._series_names = None
        super(BioformatsFile, self).__init__(*args, **kwargs)

    @property
    def series_count(self):
        if self._series_count is None:
            self._series_count = self.rdr.getSeriesCount()
        return self._series_count

    @property
    def md(self):
        if self._md is None:
            self._md = []
            _io = self.rdr.getSeries()
            for _i in range(self.series_count):
                self.rdr.setSeries(_i)
                series_md = javabridge.jutil.jdictionary_to_string_dictionary(self.rdr.getSeriesMetadata())
                self._md.append(series_md)
            self.rdr.setSeries(_io)
        return self._md

    @property
    def series_names(self):
        # return a list of series names in the file
        if self._series_names is None:
            self._series_names = []
            
            for _i in range(self.series_count):
                try:
                    self._series_names.append(self.md[_i]['Image name'])
                except(KeyError):
                    print('Image names not found, naming by integers.')
                    self._series_names.append('Image {}'.format(_i))
                    
        return self._series_names

    def __del__(self):
        release_VM()

class DataSource(XYTCDataSource):
    moduleName = 'BioformatsDataSource'
    def __init__(self, image_file, taskQueue=None, chanNum = 0, series=None):
        self.chanNum = chanNum

        if isinstance(image_file, BioformatsFile):
            self.bff = image_file
        else:
            self.filename = getFullExistingFilename(image_file)#convert relative path to full path

            print(self.filename)
            self.bff = BioformatsFile(self.filename)

        if series is not None:
            self.bff.rdr.setSeries(series)

        self.sizeX = self.bff.rdr.getSizeX()
        self.sizeY = self.bff.rdr.getSizeY()
        self.sizeZ = self.bff.rdr.getSizeZ()
        self.sizeT = self.bff.rdr.getSizeT()
        self.sizeC = self.bff.rdr.getSizeC()
    
        #self.shape = [self.sizeX, self.sizeY, self.sizeZ*self.sizeT, self.sizeC]
        #axisOrder = self.bff.rdr.getDimensionOrder()

        #sh = {'X' : self.sizeX, 'Y':self.sizeY, 'Z':self.sizeZ, 'T':self.sizeT, 'C':self.sizeT}
        
        self.additionalDims = 'TC'

    def getSlice(self, ind):
        #self.im.seek(ind)
        #ima = np.array(im.getdata()).newbyteorder(self.endedness)
        #return ima.reshape((self.im.size[1], self.im.size[0]))
        #return self.data[:,:,ind]
        c = np.floor(ind/(self.sizeZ*self.sizeT))
        t = np.floor(ind/(self.sizeZ))%self.sizeT
        z = ind % self.sizeZ
        
        res =  self.bff.read(c, z, t, rescale=False).squeeze()
        if res.ndim == 3:
            print(res.shape)
            print(self.chanNum)
            res = res[:,:,self.chanNum]
            #res = res[0,self.chanNum, :,:].squeeze()
        #print res.shape
        return res

    def getSliceShape(self):
        return (self.sizeY, self.sizeX)

    def getNumSlices(self):
        return self.sizeC*self.sizeT*self.sizeZ

    def getEvents(self):
        return []

    def release(self):
        #self.im.close()
        pass

    def reloadData(self):
        pass
    
    # Moved to BioformatsFile
    # def __del__(self):
    #     release_VM()
