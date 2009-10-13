#!/usr/bin/python

##################
# KdfSeqToHdf5.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/python

import read_kdf
import tables
import os
import sys
import numpy

from PYME.Analysis import MetaData

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)


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

        outF = tables.openFile(outFile, 'w')

        filt = tables.Filters(complevel, complib, shuffle=True)

        imageData = outF.createEArray(outF.root, 'ImageData', tables.UInt16Atom(), (0,xSize,ySize), filters=filt, expectedrows=nFrames)

        for i in range(nFrames):
            d1 = numpy.fromfile(f1, '>u2', xSize*ySize)
            imageData.append(d1.reshape(1, xSize, ySize))

        f1.close()

        hdh = MetaData.HDFMDHandler(outF, MetaData.TIRFDefault)

        if not pixelsize == None:
            hdh.setEntry('voxelsize.x', pixelsize)
            hdh.setEntry('voxelsize.y', pixelsize)

        outEvents = outF.createTable(outF.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))

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
        raise 'Usage: KdfStacktoHdf5.py inFile [outFile]'

#    if not (inDir[-1] == os.sep):
#        inDir += os.sep #append a / to directroy name if necessary

    if (len(sys.argv) == 2): #generate an output file name
        outFile = inDir[:-1] + '.h5'

    convertFile(inDir, outFile, frameSize=[256, 256], pixelsize = pixelsize)


