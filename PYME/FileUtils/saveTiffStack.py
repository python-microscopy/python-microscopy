#!/usr/bin/python

##################
# saveTiffStack.py
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

from PIL import Image
from PYME.misc import TiffImagePlugin #monkey patch PIL with improved tiff support from Priithon
import subprocess

def writeTiff(im, outfile):
    command = ["tiffcp"]
    # add options here, if any (e.g. for compression)

    #im = im.astype('uint16')
    im = im.astype('>u2').astype('<u2')

    for i in range(im.shape[2]):
        framefile = "/tmp/frame%d.tif" % i

        Image.fromarray(im[:,:,i].squeeze(), 'I;16').save(framefile)
        command.append(framefile)

    command.append(outfile)
    subprocess.call(command)

    # remove frame files here
    subprocess.call('rm /tmp/frame*.tif', shell=True)


#Adapted to use 3rd dimension as z from priithons 'useful.py'
def saveTiffMultipage(arr, fn, **params):
    if arr.ndim == 2:
        #fake a 3rd dimension
        #raise ValueError, "can only save 3d arrays"
        arr = arr[:,:,None]

    fp = open(fn, 'w+b')

    ifd_offsets=[]

    if arr.dtype == 'uint16':
        nptype = 'uint16'
        piltype = 'I;16'
    #elif arr.dtype == 'uint8':
    #    nptype = 'uint8'
    #    piltype = 'I;8'
    else:
        nptype = 'f'
        piltype = 'F'

    params["_debug_multipage"] = True
    for z in range(arr.shape[2]):
        ii = Image.fromarray(arr[:,:,z].astype(nptype), piltype)

        fp.seek(0,2) # go to end of file
        if z==0:
            # ref. PIL  TiffImagePlugin
            # PIL always starts the first IFD at offset 8
            ifdOffset = 8
        else:
            ifdOffset = fp.tell()

        ii.save(fp, format="TIFF", **params)

        if z>0: # correct "next" entry of previous ifd -- connect !
            ifdo = ifd_offsets[-1]
            fp.seek(ifdo)
            ifdLength = ii._debug_multipage.i16(fp.read(2))
            fp.seek(ifdLength*12,1) # go to "next" field near end of ifd
            fp.write(ii._debug_multipage.o32( ifdOffset ))

        ifd_offsets.append(ifdOffset)
    fp.close()

class TiffMP(object):
    def __init__(self, fn, **params):
        self.fp = open(fn, 'w+b')    
        self.ifd_offsets=[]
        params["_debug_multipage"] = True
        
        self.params = params
        self.z = 0
    
    def AddSlice(self, arr):
        if arr.dtype == 'uint16':
            nptype = 'uint16'
            piltype = 'I;16'
        #elif arr.dtype == 'uint8':
        #    nptype = 'uint8'
        #    piltype = 'I;8'
        else:
            nptype = 'f'
            piltype = 'F' 
        
        ii = Image.fromarray(arr.astype(nptype), piltype)

        self.fp.seek(0,2) # go to end of file
        if self.z==0:
            # ref. PIL  TiffImagePlugin
            # PIL always starts the first IFD at offset 8
            ifdOffset = 8
        else:
            ifdOffset = self.fp.tell()

        ii.save(self.fp, format="TIFF", **self.params)

        if self.z>0: # correct "next" entry of previous ifd -- connect !
            ifdo = self.ifd_offsets[-1]
            self.fp.seek(ifdo)
            ifdLength = ii._debug_multipage.i16(self.fp.read(2))
            self.fp.seek(ifdLength*12,1) # go to "next" field near end of ifd
            self.fp.write(ii._debug_multipage.o32( ifdOffset ))

        self.ifd_offsets.append(ifdOffset)
        self.z += 1
    
    def close(self):
        self.fp.close()
    