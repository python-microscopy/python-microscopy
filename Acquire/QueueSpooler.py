#!/usr/bin/python

##################
# QueueSpooler.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import tables
from PYME.Acquire import MetaDataHandler
from PYME import cSMI
import Pyro.core
import os
#import time

import PYME.Acquire.Spooler as sp
from PYME.Acquire import protocol as p

#rom PYME.Acquire import eventLog

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class EventLogger:
   def __init__(self, spool, scope, tq, queueName):
      self.spooler = spool
      self.scope = scope
      self.tq = tq
      self.queueName = queueName

   def logEvent(self, eventName, eventDescr = ''):
      if eventName == 'StartAq':
          eventDescr = '%d' % self.spooler.imNum
      self.tq.logQueueEvent(self.queueName, (eventName, eventDescr, sp.timeFcn()))
      

class Spooler(sp.Spooler):
   def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None, complevel=6, complib='zlib'):
       if 'PYME_TASKQUEUENAME' in os.environ.keys():
            taskQueueName = os.environ['PYME_TASKQUEUENAME']
       else:
            taskQueueName = 'taskQueue'
       self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)

       self.seriesName = filename

       self.tq.createQueue('HDFTaskQueue',self.seriesName, filename, frameSize = (scope.cam.GetPicWidth(), scope.cam.GetPicHeight()))

       self.md = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName)
       self.evtLogger = EventLogger(self, scope, self.tq, self.seriesName)

       sp.Spooler.__init__(self, scope, filename, acquisator, protocol, parent)
   
   def Tick(self, caller):
      self.tq.postTask(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()), self.seriesName)

      sp.Spooler.Tick(self, caller)


   
