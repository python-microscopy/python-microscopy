import os
#import logparser
import datetime
import tables
from PYME.Acquire import MetaDataHandler
from PYME import cSMI
import Pyro.core

import time

from PYME.Acquire import eventLog

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
      

class Spooler:
   def __init__(self, scope, filename, acquisator, parent=None, complevel=6, complib='zlib'):
       self.scope = scope
       self.filename=filename
       self.acq = acquisator
       self.parent = parent 
       
       #self.dirname =filename[:-4]
       #os.mkdir(self.dirname)
       
       #self.filestub = self.dirname.split(os.sep)[-1]

       self.tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')

       self.seriesName = filename

       self.tq.createQueue('HDFTaskQueue',self.seriesName, filename, frameSize = (scope.cam.GetPicWidth(), scope.cam.GetPicHeight()))

       self.md = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName)

       self.doStartLog()

       self.evtLogger = EventLogger(scope, self.tq, self.seriesName)
       eventLog.WantEventNotification.append(self.evtLogger)
       
       self.acq.WantFrameNotification.append(self.Tick)
       
       self.spoolOn = True

       self.imNum = 0

       
       
   def StopSpool(self):
       self.acq.WantFrameNotification.remove(self.Tick)
       eventLog.WantEventNotification.remove(self.evtLogger)
       self.doStopLog()
       #self.writeLog()
       
       self.spoolOn = False
   
   def Tick(self, caller):
      #fn = self.dirname + os.sep + self.filestub +'%05d.kdf' % self.imNum
      #caller.ds.SaveToFile(fn.encode())

      self.tq.postTask(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()), self.seriesName)
      self.imNum += 1
      if not self.parent == None:
         self.parent.Tick()

   def doStartLog(self):
      #md = self.h5File.createGroup(self.h5File.root, 'MetaData')

      dt = datetime.datetime.now()
        
      self.dtStart = dt

      
      self.md.setEntry('StartTime', time.time())
      

      #loop over all providers of metadata
      for mdgen in MetaDataHandler.provideStartMetadata:
         mdgen(self.md)
      
   
  

   def doStopLog(self):
        
        dt = datetime.datetime.now()
        
        self.md.setEntry('EndTime', time.time())
        self.md.setEntry('SpoolingFinished', True)

        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.md)
        
   def writeLog_(self):
        lw = logparser.logwriter()
        s = lw.write(self.log)
        log_f = file(self.filename, 'w')
        log_f.write(s)
        log_f.close()
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
