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

from PYME.Analysis import MetaData

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)


def convertFile(pathToData, outFile, pixelsize=None, complib='zlib', complevel=9):

        #seriesName = pathToData.split(os.sep)[-2]


        #fnl = os.listdir(pathToData)
        #fnl2 = [pathToData + f for f in fnl]

        f1 = read_kdf.ReadKdfData(pathToData).squeeze()

        xSize, ySize = f1.shape[0:2]

        outF = tables.openFile(outFile, 'w')

        filt = tables.Filters(complevel, complib, shuffle=True)

        imageData = outF.createEArray(outF.root, 'ImageData', tables.UInt16Atom(), (0,xSize,ySize), filters=filt, expectedrows=f1.shape[2])

        for i in range(f1.shape[2]):
            imageData.append(f1[:,:,i].reshape(1, xSize, ySize))

        hdh = MetaData.HDFMDHandler(outF, MetaData.PCODefault)

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

    convertFile(inDir, outFile, pixelsize = pixelsize)


