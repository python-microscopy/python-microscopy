#!/usr/bin/python

##################
# QueueSpooler.py
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

import tables
from PYME.Acquire import MetaDataHandler
#from PYME import cSMI
#import Pyro.core
import os
import time

import PYME.Acquire.Spooler as sp
from PYME.Acquire import protocol as p
from PYME.FileUtils import fileID, nameUtils
from PYME.ParallelTasks.relativeFiles import getRelFilename

import httplib
import cPickle as pickle
import threading
import Queue

#rom PYME.Acquire import eventLog

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class EventLogger:
   def __init__(self, spool, scope):
      self.spooler = spool
      self.scope = scope

   def logEvent(self, eventName, eventDescr = ''):
      if eventName == 'StartAq':
          eventDescr = '%d' % self.spooler.imNum
      self.spooler._post('NEWEVENT', (eventName, eventDescr, sp.timeFcn()))
      
class HttpSpoolMDHandler(MetaDataHandler.MDHandlerBase):
    def __init__(self, spooler, mdToCopy=None):
        self.spooler = spooler
        self.cache = {}
        
        if not mdToCopy == None:
            self.copyEntriesFrom(mdToCopy)

    def setEntry(self,entryName, value):
        self.spooler._post('METADATAENTRY', (entryName, value))
    
    def getEntry(self,entryName):
        return self.cache[entryName]

    def getEntryNames(self):
        return self.cache.keys()

SERVERNAME='127.0.0.1:8080'      

class Spooler(sp.Spooler):
    def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None, complevel=2, complib='zlib'):
        #       if 'PYME_TASKQUEUENAME' in os.environ.keys():
        #            taskQueueName = os.environ['PYME_TASKQUEUENAME']
        #       else:
        #            taskQueueName = 'taskQueue'
        #from PYME.misc.computerName import GetComputerName
        #compName = GetComputerName()
        
        #taskQueueName = 'TaskQueues.%s' % compName
        
        #self.tq = Pyro.core.getProxyForURI('PYRONAME://' + taskQueueName)
        #self.tq._setOneway(['postTask', 'postTasks', 'addQueueEvents', 'setQueueMetaData', 'logQueueEvent'])
        filename = filename[len(nameUtils.datadir):]
        
        
        self.seriesName = '/'.join(filename.split(os.path.sep))
        print filename, self.seriesName
        self.buffer = []
        self.buflen = 30
        
        
        
        self.postQueue = Queue.Queue()
        self.dPoll = True
        
        self.pollThread = threading.Thread(target=self._queuePoll)
        self.pollThread.start()
        
        #self.tq.createQueue('HDFTaskQueue',self.seriesName, filename, frameSize = (scope.cam.GetPicWidth(), scope.cam.GetPicHeight()), complevel=complevel, complib=complib)
        
        self.md = HttpSpoolMDHandler(self)
        self.evtLogger = EventLogger(self, scope)
        
        
        sp.Spooler.__init__(self, scope, filename, acquisator, protocol, parent)
        
    def _queuePoll(self):
        self.conn = httplib.HTTPConnection(SERVERNAME, timeout=5)
        while self.dPoll:            
            ur, data = self.postQueue.get()
            print ur
            self.conn.request('POST', ur, pickle.dumps(data))
            
            resp = self.conn.getresponse()
            print resp.status, resp.reason
            time.sleep(.1)
    
    def _post(self, ursufix, data):
       self.postQueue.put(('%s/%s' % (self.seriesName, ursufix), data))
        
    def getURL(self):
        return 'http://' + SERVERNAME + self.seriesName
    
    def StopSpool(self):
        self.dPoll = False
        sp.Spooler.StopSpool(self)
        
        
    def Tick(self, caller):
      #self.tq.postTask(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()), self.seriesName)
      self.buffer.append(caller.dsa.reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()).copy())

      if self.imNum == 0: #first frame
          self.md.setEntry('imageID', fileID.genFrameID(self.buffer[-1].squeeze()))

      if len(self.buffer) >= self.buflen:
          self.FlushBuffer()

      sp.Spooler.Tick(self, caller)
      
    def FlushBuffer(self):
      t1 = time.time()
      self._post('NEWFRAMES', self.buffer)
      #print time.time() -t1
      self.buffer = []
     



   
