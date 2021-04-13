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

from __future__ import print_function
import tables
from PYME.IO import MetaDataHandler

#import Pyro.core
import os
import time

import PYME.Acquire.Spooler as sp
#from PYME.Acquire import protocol as p
from PYME.IO.FileUtils import fileID, nameUtils
#from PYME.IO.FileUtils.nameUtils import getRelFilename


try:
    # noinspection PyCompatibility
    import httplib
except ImportError:
    #py3
    import http.client as httplib

try:
    # noinspection PyCompatibility
    import cPickle as pickle
except ImportError:
    #py3
    import pickle
    
import threading

try:
    # noinspection PyCompatibility
    import Queue
except ImportError:
    #py3
    import queue as Queue
    
import requests

#rom PYME.Acquire import eventLog

class SpoolEvent(tables.IsDescription):
   EventName = tables.StringCol(32)
   Time = tables.Time64Col()
   EventDescr = tables.StringCol(256)

class EventLogger:
   def __init__(self, spool):#), scope):
      self.spooler = spool
      #self.scope = scope

   def logEvent(self, eventName, eventDescr = '', timestamp = None):
      if eventName == 'StartAq':
          eventDescr = '%d' % self.spooler.imNum

      if timestamp is None:
          timestamp = sp.timeFcn()
      self.spooler._post('NEWEVENT', (eventName, eventDescr, timestamp))
      
class HttpSpoolMDHandler(MetaDataHandler.MDHandlerBase):
    def __init__(self, spooler, mdToCopy=None):
        self.spooler = spooler
        self.cache = {}
        
        if not mdToCopy is None:
            self.copyEntriesFrom(mdToCopy)

    def setEntry(self,entryName, value):
        self.spooler._post('METADATAENTRY', (entryName, value))
    
    def getEntry(self,entryName):
        return self.cache[entryName]

    def getEntryNames(self):
        return self.cache.keys()
        
    def copyEntriesFrom(self, mdToCopy):            
        self.spooler._post('METADATA', MetaDataHandler.NestedClassMDHandler(mdToCopy))

SERVERNAME='127.0.0.1:8080'      

class Spooler(sp.Spooler):
    def __init__(self, filename, frameSource, frameShape, **kwargs):
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
        #print filename, self.seriesName
        self.buffer = []
        self.buflen = 30
        
        
        
        self.postQueue = Queue.Queue()
        self.dPoll = True
        
        self.pollThread = threading.Thread(target=self._queuePoll)
        self.pollThread.start()
        
        #self.tq.createQueue('HDFTaskQueue',self.seriesName, filename, frameSize = (scope.cam.GetPicWidth(), scope.cam.GetPicHeight()), complevel=complevel, complib=complib)
        
        self.md = HttpSpoolMDHandler(self)
        self.evtLogger = EventLogger(self)
        
        
        sp.Spooler.__init__(self, filename, frameSource, **kwargs)
        
    def __queuePoll(self):
        self.conn = httplib.HTTPConnection(SERVERNAME, timeout=5)
        while self.dPoll:            
            ur, data = self.postQueue.get()
            #print(repr(ur))
            #try:
            self.conn.request('POST', ur.encode(), pickle.dumps(data, 2), {"Connection":"keep-alive"})
            
            resp = self.conn.getresponse()
            #print resp.status, resp.reason
            #except UnicodeDecodeError:
            #    print self.conn._buffer
            time.sleep(.1)
            
            
    def _queuePoll(self):
        #self.conn = 
        while self.dPoll:            
            ur, data = self.postQueue.get()
            #print repr(ur)
            #conn = httplib.HTTPConnection(SERVERNAME, timeout=15)
            #print 'hc'
            #conn.request('POST', ur.encode(), pickle.dumps(data, 2))#, {"Connection":"keep-alive"})
            #print 'rq'
            #resp = conn.getresponse()
            #print 'rp'
            #conn.close()
            
            r = requests.post('http://' + SERVERNAME + ur.encode(), pickle.dumps(data, 2))
            
            #print r.status_code
                
            #print resp.status, resp.reason
            #except UnicodeDecodeError:
            #    print self.conn._buffer
            time.sleep(.1)
            
    def ___queuePoll(self):
        #self.conn = 
        while self.dPoll:            
            ur, data = self.postQueue.get()
            #print repr(ur)
            conn = httplib.HTTPConnection(SERVERNAME, timeout=15)
            #print 'hc'
            conn.request('POST', ur.encode(), pickle.dumps(data, 2))#, {"Connection":"keep-alive"})
            #print 'rq'
            resp = conn.getresponse()
            #print 'rp'
            #conn.close()
                
            #print resp.status, resp.reason
            #except UnicodeDecodeError:
            #    print self.conn._buffer
            time.sleep(.1)
    
    def _post(self, ursufix, data):
       self.postQueue.put(('%s/%s' % (self.seriesName, ursufix), data))
        
    def getURL(self):
        return 'http://' + SERVERNAME + self.seriesName
    
    def StopSpool(self):
        self.dPoll = False
        sp.Spooler.StopSpool(self)
        
        
    def OnFrame(self, sender, frameData, **kwargs):
      self.buffer.append(frameData.reshape(1,frameData.shape[0],frameData.shape[1]).copy())

      if self.imNum == 0: #first frame
          self.md.setEntry('imageID', fileID.genFrameID(self.buffer[-1].squeeze()))
          #pass

      if len(self.buffer) >= self.buflen:
          self.FlushBuffer()

      sp.Spooler.OnFrame(self)
      
    def FlushBuffer(self):
      t1 = time.time()
      self._post('NEWFRAMES', self.buffer)
      #print time.time() -t1
      self.buffer = []
     



   
