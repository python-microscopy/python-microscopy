#!/usr/bin/python

##################
# KdfSeqToHdf5.py
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
import numpy
from PYME.IO.events import SpoolEvent
from PYME.Analysis import MetaData


def convertFile(pathToData, outFile, frameSize = [256,256], pixelsize=None, complib='zlib', complevel=9):

        #seriesName = pathToData.split(os.sep)[-2]


        #fnl = os.listdir(pathToData)
        #fnl2 = [pathToData + f for f in fnl]

        #f1 = read_kdf.ReadKdfData(pathToData).squeeze()
        
        xSize, ySize = frameSize

        f1 = open(pathToData, 'rb')

        #detect file length
        f1.seek(0,2) #seek to end
        fLength = f1.tell()
        f1.seek(0) #back to begining

        nFrames = fLength/(2*xSize*ySize)

        outF = tables.open_file(outFile, 'w')

        filt = tables.Filters(complevel, complib, shuffle=True)

        imageData = outF.create_earray(outF.root, 'ImageData', tables.UInt16Atom(), (0,xSize,ySize), filters=filt, expectedrows=nFrames)

        for i in range(nFrames):
            d1 = numpy.fromfile(f1, '>u2', xSize*ySize) >> 4
            imageData.append(d1.reshape(1, xSize, ySize))
            if i % 100 == 0:
                print(('%d of %d frames' % (i, nFrames)))

        f1.close()

        hdh = MetaData.HDFMDHandler(outF, MetaData.TIRFDefault)

        if not pixelsize is None:
            hdh.setEntry('voxelsize.x', pixelsize)
            hdh.setEntry('voxelsize.y', pixelsize)

        outEvents = outF.create_table(outF.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))

        outF.flush()
        outF.close()



if __name__ == '__main__':
    pixelsize=None
    if (len(sys.argv) == 4):
        inDir = sys.argv[1]
        outFile = sys.argv[2]
        pixelsize = float(sys.argv[3])
    elif (len(sys.argv) == 3):
        inDir = sys.argv[1]
        outFile = sys.argv[2]
    elif (len(sys.argv) == 2):
        inDir = sys.argv[1]
    else: 
        raise RuntimeError('Usage: KdfStacktoHdf5.py inFile [outFile]')

#    if not (inDir[-1] == os.sep):
#        inDir += os.sep #append a / to directroy name if necessary

    if (len(sys.argv) == 2): #generate an output file name
        outFile = inDir[:-1] + '.h5'

    convertFile(inDir, outFile, frameSize=[256, 256], pixelsize = pixelsize)


