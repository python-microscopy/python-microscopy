#!/usr/bin/python

##################
# TiffSeqToHdf5.py
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

#!/usr/bin/python

#import read_kdf
import tables
import os
import sys

from PIL import Image
from PYME.IO.events import SpoolEvent
from numpy import array



def convertFiles(pathToData, outFile, complib='zlib', complevel=9):

        seriesName = pathToData.split(os.sep)[-2]


        fnl = os.listdir(pathToData)
        fnl.sort()
        fnl2 = [pathToData + f for f in fnl]

        #f1 = read_kdf.ReadKdfData(fnl2[0])

        f1 = array(Image.open(fnl2[0])).newbyteorder('BE')

        xSize, ySize = f1.shape[0:2]

        outF = tables.open_file(outFile, 'w')

        filt = tables.Filters(complevel, complib, shuffle=True)

        imageData = outF.create_earray(outF.root, 'ImageData', tables.UInt16Atom(), (0,xSize,ySize), filters=filt, expectedrows=len(fnl2))

        for fn in fnl2:
            #print fn
            imageData.append(array(Image.open(fn)).newbyteorder('BE').reshape(1, xSize, ySize))

        outEvents = outF.create_table(outF.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))

        outF.flush()
        outF.close()



if __name__ == '__main__':
     
    if (len(sys.argv) == 3):
        inDir = sys.argv[1]
        outFile = sys.argv[2]
    elif (len(sys.argv) == 2):
        inDir = sys.argv[1]
    else: 
        raise RuntimeError('Usage: TiffSeqtoHdf5.py inDir [outFile]')

    if not (inDir[-1] == os.sep):
        inDir += os.sep #append a / to directroy name if necessary

    if (len(sys.argv) == 2): #generate an output file name
        outFile = inDir[:-1] + '.h5'

    convertFiles(inDir, outFile)


