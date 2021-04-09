#!/usr/bin/python

##################
# HDFSpooler.py
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

import datetime
import tables
from PYME.IO import MetaDataHandler
from PYME.IO.events import HDFEventLogger

import time

import PYME.Acquire.Spooler as sp

from PYME.IO.FileUtils import fileID

class Spooler(sp.Spooler):
    """Responsible for the mechanics of spooling to a pytables/hdf file.
    """
    def __init__(self, filename, frameSource, frameShape, complevel=6, complib='zlib', **kwargs):
        self.h5File = tables.open_file(filename, 'w')
           
        filt = tables.Filters(complevel, complib, shuffle=True)
        
        self.imageData = self.h5File.create_earray(self.h5File.root, 'ImageData', tables.UInt16Atom(), (0,frameShape[0],frameShape[1]), filters=filt)
        self.md = MetaDataHandler.HDFMDHandler(self.h5File)
        self.evtLogger = HDFEventLogger(self, self.h5File)
        
        sp.Spooler.__init__(self, filename, frameSource, **kwargs)

    def finalise(self):
        """ close files"""
           
        self.h5File.flush()
        self.h5File.close()
        
    def OnFrame(self, sender, frameData, **kwargs):
        """Called on each frame"""

        if not self.watchingFrames:
            # drop frames if we've already stopped spooling. TODO - do we also need to check if disconnect works?
            return
        
        #print 'f'
        if frameData.shape[0] == 1:
            self.imageData.append(frameData)
        else:
            self.imageData.append(frameData.reshape(1,frameData.shape[0],frameData.shape[1]))
        self.h5File.flush()
        if self.imNum == 0: #first frame
            self.md.setEntry('imageID', fileID.genFrameID(self.imageData[0,:,:]))
            
        sp.Spooler.OnFrame(self)
        
    def __del__(self):
        if self.spoolOn:
            self.StopSpool()
