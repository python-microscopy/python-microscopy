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
from PYME.Acquire import MetaDataHandler
#from PYME import cSMI

#import time

#from PYME.Acquire import eventLog
import PYME.Acquire.Spooler as sp
from PYME.Acquire import protocol as p

from PYME.FileUtils import fileID

class SpoolEvent(tables.IsDescription):
    '''Pytables description for Events table in spooled dataset'''
    EventName = tables.StringCol(32)
    Time = tables.Time64Col()
    EventDescr = tables.StringCol(256)

class EventLogger:
    '''Event logging backend for hdf/pytables data storage
        
    Parameters
    ----------
    spool : instance of HDFSpooler.Spooler
        The spooler to ascociate this logger with
    
    scope : PYME.Acquire.microscope.microscope instance
        The current microscope object
    
    hdf5File : pytables hdf file 
        The open HDF5 file to write to
    '''
    def __init__(self, spool, scope, hdf5File):
      '''Create a new Events table.
      
      
      '''
      self.spooler = spool
      self.scope = scope
      self.hdf5File = hdf5File
    
      self.evts = self.hdf5File.createTable(hdf5File.root, 'Events', SpoolEvent)

    def logEvent(self, eventName, eventDescr = ''):
        '''Log an event.
          
        Parameters
        ----------
        eventName : string
            short event name - < 32 chars and should be shared by events of the
            same type.
        eventDescr : string
            description of the event - additional, even specific information
            packaged as a string (<255 chars). This is commonly used to store 
            parameters - e.g. z positions, and should be both human readable and 
            easily parsed.
        
        
        In addition to the name and description, timing information is recorded
        for each event.
        '''
        if eventName == 'StartAq':
            eventDescr = '%d' % self.spooler.imNum
              
        ev = self.evts.row
        
        ev['EventName'] = eventName
        ev['EventDescr'] = eventDescr
        ev['Time'] = sp.timeFcn()
        
        ev.append()
        self.evts.flush()

class Spooler(sp.Spooler):
    '''Responsible for the mechanics of spooling to a pytables/hdf file.
    '''
    def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None, complevel=6, complib='zlib'):
        self.h5File = tables.openFile(filename, 'w')
           
        filt = tables.Filters(complevel, complib, shuffle=True)
        
        self.imageData = self.h5File.createEArray(self.h5File.root, 'ImageData', tables.UInt16Atom(), (0,scope.cam.GetPicWidth(),scope.cam.GetPicHeight()), filters=filt)
        self.md = MetaDataHandler.HDFMDHandler(self.h5File)
        self.evtLogger = EventLogger(self, scope, self.h5File)
        
        sp.Spooler.__init__(self, scope, filename, acquisator, protocol, parent)

    def StopSpool(self):
        '''Stop spooling and close file'''
        sp.Spooler.StopSpool(self)
           
        self.h5File.flush()
        self.h5File.close()
        
    def Tick(self, caller):
        '''Called on each frame'''
        self.imageData.append(caller.dsa.reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()))
        self.h5File.flush()
        if self.imNum == 0: #first frame
            self.md.setEntry('imageID', fileID.genFrameID(self.imageData[0,:,:]))
            
        sp.Spooler.Tick(self, caller)
        
    def __del__(self):
        if self.spoolOn:
            self.StopSpool()
