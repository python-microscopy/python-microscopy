#!/usr/bin/python

##################
# Spooler.py
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

import os
#import logparser
import datetime

from PYME.Acquire import MetaDataHandler
from PYME import cSMI

try:
    from PYME.Acquire import sampleInformation
except:
    sampleInformation= None

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

       self.imNum = 0

       #if we've got a fake camera - the cycle time will be wrong - fake our time sig to make up for this
       if scope.cam.__class__.__name__ == 'FakeCamera':
           timeFcn = self.fakeTime

       self.protocol.Init(self)
       
       self.acq.WantFrameNotification.append(self.Tick)
       self.spoolOn = True

       
       
   def StopSpool(self):
       self.acq.WantFrameNotification.remove(self.Tick)

       try:
           self.protocol.OnFinish()#this may still cause events
           self.FlushBuffer()
           self.doStopLog()
       except:
           import traceback
           traceback.print_exc()

       eventLog.WantEventNotification.remove(self.evtLogger)
       
       self.spoolOn = False

   def Tick(self, caller):
        self.imNum += 1
        if not self.parent == None:
            self.parent.Tick()
        self.protocol.OnFrame(self.imNum)

        if self.imNum == 2 and sampleInformation and sampleInformation.currentSlide[0]: #have first frame and should thus have an imageID
            sampleInformation.createImage(self.md, sampleInformation.currentSlide[0])

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

   def FlushBuffer(self):
       pass
        
        
   def __del__(self):
        if self.spoolOn:
            self.StopSpool()
