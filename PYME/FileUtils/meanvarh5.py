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
import numpy as np

from PYME.Acquire import MetaDataHandler
from PYME.Analysis.DataSources import HDFDataSource

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

# usage: cd ~/Downloads
# usage: import PYME.FileUtils.meanvarh5 as mv
# usage: mv.meanvarh5(do.ds.datasource,mdv.mdh,do.ds.datasource.h5Filename,'1kdf10msA.h5',0,500)
def meanvarh5(dataSource, metadata, origName, outFile, start, end):
    
    h5out = tables.openFile(outFile,'w')
#    filters=tables.Filters(complevel,complib,shuffle=True)

    nframes = end - start
    xSize, ySize = dataSource.getSliceShape()

    m = np.zeros((xSize,ySize),dtype='float64')
    for frameN in range(start,end):
        m += dataSource.getSlice(frameN)
    m = m / nframes

    v = np.zeros((xSize,ySize),dtype='float64')
    for frameN in range(start,end):
        v += (dataSource.getSlice(frameN)-m)**2
    v = v / (nframes-1)

    ims = h5out.createEArray(h5out.root,'ImageData',tables.Float32Atom(),(0,xSize,ySize), expectedrows=2)
    ims.append(m[None,:,:])
    ims.append(v[None,:,:])

    outMDH = MetaDataHandler.HDFMDHandler(h5out)

    outMDH.copyEntriesFrom(metadata)
    outMDH.setEntry('meanvar.originalFile', origName)
    outMDH.setEntry('meanvar.start', start)
    outMDH.setEntry('meanvar.end', end)

    outEvents = h5out.createTable(h5out.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))

    #copy events to results file
    evts = dataSource.getEvents()
    if len(evts) > 0:
        outEvents.append(evts)

    h5out.flush()
    h5out.close()
    
