import os
#import logparser
import datetime

from PYME.Acquire import MetaDataHandler
from PYME import cSMI

import time

from PYME.Acquire import eventLog
from PYME.Acquire import protocol as p

class EventLogger:
   def __init__(self, scope, hdf5File):
      self.scope = scope      

   def logEvent(self, eventName, eventDescr = ''):
      pass

class Spooler:
   def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None):
       self.scope = scope
       self.filename=filename
       self.acq = acquisator
       self.parent = parent
       self.protocol = protocol

       self.doStartLog()

       eventLog.WantEventNotification.append(self.evtLogger)
       
       self.acq.WantFrameNotification.append(self.Tick)

       self.imNum = 0
       self.protocol.Init()
       self.spoolOn = True

       
       
   def StopSpool(self):
       self.acq.WantFrameNotification.remove(self.Tick)
       eventLog.WantEventNotification.remove(self.evtLogger)
       self.doStopLog()
       #self.writeLog()
       
       self.spoolOn = False

   def Tick(self, caller):
        self.imNum += 1
        if not self.parent == None:
            self.parent.Tick()
        self.protocol.OnFrame(self.imNum)

   def doStartLog(self):
      dt = datetime.datetime.now()

      self.dtStart = dt
      
      self.md.setEntry('StartTime', time.time())

      #loop over all providers of metadata
      for mdgen in MetaDataHandler.provideStartMetadata:
         mdgen(self.md)
       

   def doStopLog(self):
        self.md.setEntry('EndTime', time.time())
        
        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.md)
        
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
