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
try:
    import javabridge
    import bioformats
    #javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
except:
    pass

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


from .BaseDataSource import BaseDataSource

class DataSource(BaseDataSource):
    moduleName = 'BioformatsDataSource'
    def __init__(self, filename, taskQueue=None, chanNum = 0):
        self.filename = getFullExistingFilename(filename)#convert relative path to full path
        self.chanNum = chanNum
        
        #self.data = readTiff.read3DTiff(self.filename)

        #self.im = Image.open(filename)

        #self.im.seek(0)

        #PIL's endedness support is subtly broken - try to fix it
        #NB this is untested for floating point tiffs
        #self.endedness = 'LE'
        #if self.im.ifd.prefix =='MM':
        #    self.endedness = 'BE'

        #to find the number of images we have to loop over them all
        #this is obviously not ideal as PIL loads the image data into memory for each
        #slice and this is going to represent a huge performance penalty for large stacks
        #should still let them be opened without having all images in memory at once though
        #self.numSlices = self.im.tell()
        
        #try:
        #    while True:
        #        self.numSlices += 1
        #        self.im.seek(self.numSlices)
                
        #except EOFError:
        #    pass

        print((self.filename))
        
        #tf = tifffile.TIFFfile(self.filename)
        ensure_VM()
        self.bff = bioformats.ImageReader(filename)
        
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
    
    def __del__(self):
        release_VM()
