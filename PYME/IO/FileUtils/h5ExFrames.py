#!/usr/bin/python

##################
# h5ExFrames.py
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

import tables
import os
import sys

from PYME.IO import MetaDataHandler
from PYME.IO.DataSources import HDFDataSource
from PYME.IO.events import SpoolEvent

def extractFrames(dataSource, metadata, origName, outFile, start, end, subsamp=1, complib='zlib', complevel=5):
    
    h5out = tables.open_file(outFile,'w')
    filters=tables.Filters(complevel,complib,shuffle=True)

    nframes = end - start
    xSize, ySize = dataSource.getSliceShape()

    ims = h5out.create_earray(h5out.root,'ImageData',tables.UInt16Atom(),(0,xSize,ySize), filters=filters, expectedrows=nframes)
    for frameN in range(start,end, subsamp):
        im = dataSource.getSlice(frameN)[None, :,:]
        for fN in range(frameN+1, frameN+subsamp):
            im += dataSource.getSlice(fN)[None, :,:]
        ims.append(im)
        ims.flush()

    outMDH = MetaDataHandler.HDFMDHandler(h5out)

    outMDH.copyEntriesFrom(metadata)
    outMDH.setEntry('cropping.originalFile', origName)
    outMDH.setEntry('cropping.start', start)
    outMDH.setEntry('cropping.end', end)
    outMDH.setEntry('cropping.averaging', subsamp)

    if 'Camera.ADOffset' in metadata.getEntryNames():
        outMDH.setEntry('Camera.ADOffset', subsamp*metadata.getEntry('Camera.ADOffset'))


    outEvents = h5out.create_table(h5out.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))

    #copy events to results file
    evts = dataSource.getEvents()
    if len(evts) > 0:
        outEvents.append(evts)


    h5out.flush()
    h5out.close()
    



def extractFramesF(inFile, outFile, start, end, complib='zlib', complevel=9):
    h5in = HDFDataSource(inFile)

    md = MetaDataHandler.HDFMDHandler(h5in.h5File)

    extractFrames(h5in, md, h5in.h5File.filename, outFile, start, end, complib, complevel)

    h5in.release()


if __name__ == '__main__':
     
    if (len(sys.argv) == 5):
        inFile = sys.argv[1]
        outFile = sys.argv[2]
        start = int(sys.argv[3])
        end = int(sys.argv[4])
    else: 
        raise RuntimeError('Usage: Hdf5exframes.py inFile outFile startframe endframe')

    extractFramesF(inFile, outFile, start, end)


