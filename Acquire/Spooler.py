#!/usr/bin/python

##################
# Spooler.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import os
#import logparser
import datetime

from PYME.Acquire import MetaDataHandler
from PYME import cSMI

import time

global timeFcn
timeFcn = time.time

from PYME.Acquire import eventLog
from PYME.Acquire import protocol as p

class EventLogger:
   def __init__(self, scope, hdf5File):
      self.scope = scope      

   def logEvent(self, eventName, eventDescr = ''):
      pass

class Spooler:
   def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None):
       global timeFcn
       self.scope = scope
       self.filename=filename
       self.acq = acquisator
       self.parent = parent
       self.protocol = protocol

       self.doStartLog()

       eventLog.WantEventNotification.append(self.evtLogger)
       
       self.acq.WantFrameNotification.append(self.Tick)

       self.imNum = 0

       #if we've got a fake camera - the cycle time will be wrong - fake our time sig to make up for this
       if scope.cam.__class__.__name__ == 'FakeCamera':
           timeFcn = self.fakeTime

       self.protocol.Init(self)
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

      self.tStart = time.time()
      
      self.md.setEntry('StartTime', self.tStart)

      #loop over all providers of metadata
      for mdgen in MetaDataHandler.provideStartMetadata:
         mdgen(self.md)
       

   def doStopLog(self):
        self.md.setEntry('EndTime', time.time())
        
        #loop over all providers of metadata
        for mdgen in MetaDataHandler.provideStopMetadata:
           mdgen(self.md)

   def fakeTime(self):
       return self.tStart + self.imNum*self.scope.cam.GetIntegTime()
        
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
