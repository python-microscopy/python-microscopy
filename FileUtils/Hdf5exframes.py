#!/usr/bin/python

import tables
import os
import sys





def extractFrames(inFile, outFile, start, end, complib='zlib', complevel=9):

    h5in = tables.openFile(inFile)

    h5out = tables.openFile(outFile,'w')

    filters=tables.Filters(complevel,complib,shuffle=True)

#     data = h5in.root.ImageData[start:end,:,:]

#     ims = h5out.createCArray(h5out.root,'ImageData',tables.UInt16Atom(),data.shape,filters=filters)
#     ims[:,:,:] = data

# this one might have the smaller memory footprint
    nframes = end - start
    xSize, ySize = h5in.root.ImageData.shape[1:3]
    ims = h5out.createEArray(h5out.root,'ImageData',tables.UInt16Atom(),(0,xSize,ySize), filters=filters, expectedrows=nframes)
    for frame in range(start,end):
	ims.append(h5in.root.ImageData[frame:frame+1,:,:])

    h5out.flush()
    h5out.close()
    h5in.close()


if __name__ == '__main__':
     
    if (len(sys.argv) == 5):
        inFile = sys.argv[1]
        outFile = sys.argv[2]
	start = int(sys.argv[3])
	end = int(sys.argv[4])
    else: 
        raise 'Usage: Hdf5exframes.py inFile outFile startframe endframe'

    extractFrames(inFile, outFile, start, end)


