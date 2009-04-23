import tables
from PYME.Acquire import MetaDataHandler
from PYME import cSMI
import Pyro.core

import PYME.Acquire.Spooler as sp
from PYME.Acquire import protocol as p

#rom PYME.Acquire import eventLog

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class EventLogger:
   def __init__(self, scope, tq, queueName):
      self.scope = scope
      self.tq = tq
      self.queueName = queueName

   def logEvent(self, eventName, eventDescr = ''):
      self.tq.logQueueEvent(self,queueName, (eventName, eventDescr, time.time()))
      

class Spooler(sp.Spooler):
   def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None, complevel=6, complib='zlib'):
       self.tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

       self.seriesName = filename

       self.tq.createQueue('HDFTaskQueue',self.seriesName, filename, frameSize = (scope.cam.GetPicWidth(), scope.cam.GetPicHeight()))

       self.md = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName)
       self.evtLogger = EventLogger(scope, self.tq, self.seriesName)

       sp.Spooler.__init__(self, scope, filename, acquisator, protocol, parent)

   
   def Tick(self, caller):
      self.tq.postTask(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()), self.seriesName)

      sp.Spooler.Tick(self, caller)


   
