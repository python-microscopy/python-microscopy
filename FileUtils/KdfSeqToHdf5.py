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





def convertFiles(pathToData, outFile, complib='zlib', complevel=9):

        seriesName = pathToData.split(os.sep)[-2]


        fnl = os.listdir(pathToData)
        fnl2 = [pathToData + f for f in fnl]

        f1 = read_kdf.ReadKdfData(fnl2[0])

        xSize, ySize = f1.shape[0:2]

        outF = tables.openFile(outFile, 'w')

        filt = tables.Filters(complevel, complib, shuffle=True)

        imageData = outF.createEArray(outF.root, 'ImageData', tables.UInt16Atom(), (0,xSize,ySize), filters=filt, expectedrows=len(fnl2))

        for fn in fnl2:
            imageData.append(read_kdf.ReadKdfData(fn).reshape(1, xSize, ySize))

        outF.flush()
        outF.close()



if __name__ == '__main__':
     
    if (len(sys.argv) == 3):
        inDir = sys.argv[1]
        outFile = sys.argv[2]
    elif (len(sys.argv) == 2):
        inDir = sys.argv[1]
    else: 
        raise 'Usage: KdfSeqtoHdf5.py inDir [outFile]'

    if not (inDir[-1] == os.sep):
        inDir += os.sep #append a / to directroy name if necessary

    if (len(sys.argv) == 2): #generate an output file name
        outFile = inDir[:-1] + '.h5'

    convertFiles(inDir, outFile)


