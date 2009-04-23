import datetime
import tables
from PYME.Acquire import MetaDataHandler
from PYME import cSMI

import time

#from PYME.Acquire import eventLog
import PYME.Acquire.Spooler as sp
from PYME.Acquire import protocol as p

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class EventLogger:
   def __init__(self, scope, hdf5File):
      self.scope = scope
      self.hdf5File = hdf5File

      self.evts = self.hdf5File.createTable(hdf5File.root, 'Events', SpoolEvent)

   def logEvent(self, eventName, eventDescr = ''):
      ev = self.evts.row

      ev['EventName'] = eventName
      ev['EventDescr'] = eventDescr
      ev['Time'] = time.time()

      ev.append()
      self.evts.flush()

class Spooler(sp.Spooler):
   def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None, complevel=6, complib='zlib'):
       self.h5File = tables.openFile(filename, 'w')
       
       filt = tables.Filters(complevel, complib, shuffle=True)

       self.imageData = self.h5File.createEArray(self.h5File.root, 'ImageData', tables.UInt16Atom(), (0,scope.cam.GetPicWidth(),scope.cam.GetPicHeight()), filters=filt)
       self.md = MetaDataHandler.HDFMDHandler(self.h5File)
       self.evtLogger = EventLogger(scope, self.h5File)

       sp.Spooler.__init__(self, scope, filename, acquisator, protocol, parent)

       
       
   def StopSpool(self):
       sp.Spooler.StopSpool(self)
       
       self.h5File.flush()
       self.h5File.close()
   
   def Tick(self, caller):      
      self.imageData.append(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()))
      self.h5File.flush()

      sp.Spooler.Tick(self, caller)
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
