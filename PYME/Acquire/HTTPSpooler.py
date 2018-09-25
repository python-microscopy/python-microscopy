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
from PYME.IO import MetaDataHandler

#import Pyro.core
import os
import time

import PYME.Acquire.Spooler as sp
#from PYME.Acquire import protocol as p
#from PYME.IO.FileUtils import fileID, nameUtils
#from PYME.ParallelTasks.relativeFiles import getRelFilename

import threading

try:
    # noinspection PyCompatibility
    import Queue
except ImportError:
    #py3
    import queue as Queue

from PYME.IO import clusterIO
from PYME.IO import PZFFormat

import numpy as np
import random

import json

import logging
logger = logging.getLogger(__name__)

class EventLogger:
    def __init__(self, spool):#, scope):
        self.spooler = spool
        #self.scope = scope
          
        self._events = []
    
    def logEvent(self, eventName, eventDescr = '', timestamp=None):
        if eventName == 'StartAq' and eventDescr == '':
            eventDescr = '%d' % self.spooler.imNum

        if timestamp is None:
            timestamp = sp.timeFcn()
            
        self._events.append((eventName, eventDescr, timestamp))
        
    def to_JSON(self):
        return json.dumps(self._events)
          

CLUSTERID=''  

def genSequenceID(filename=''):
    return  int(time.time()) & random.randint(0, 2**31) << 31 
    

    

def getReducedFilename(filename):
    #rname = filename[len(nameUtils.datadir):]
        
    sname = '/'.join(filename.split(os.path.sep))
    if sname.startswith('/'):
        sname = sname[1:]
    
    return sname
    
def exists(seriesName):
    return clusterIO.exists(getReducedFilename(seriesName) + '/')

#Push data to cluster from multiple threads simultaeneously to hide IO latency
#of each individual node. Not sure what the best number is here - currenty set
#a "safe" maximum number of nodes that could access data
NUM_POLL_THREADS = 10

defaultCompSettings = {
    'compression' : PZFFormat.DATA_COMP_HUFFCODE,
    'quantization' : PZFFormat.DATA_QUANT_SQRT,
    'quantizationOffset' : 0.0,
    'quantizationScale' : 1.0
}

class Spooler(sp.Spooler):
    def __init__(self, filename, frameSource, frameShape, **kwargs):
        
        #filename = filename[len(nameUtils.datadir):]
        
        #filename, 
        self.seriesName = getReducedFilename(filename)
        
        #self.seriesName = '/'.join(filename.split(os.path.sep))
        #if self.seriesName.startswith('/'):
        #    self.seriesName = self.seriesName[1:]
        #print filename, self.seriesName
        self._aggregate_h5 = kwargs.get('aggregate_h5', False)
        
        self.clusterFilter = kwargs.get('serverfilter', CLUSTERID)
        self._buffer = []
        
        self.buflen = 50
        
        self._postQueue = Queue.Queue()
        self._dPoll = True
        self._lock = threading.Lock()
        
        self._last_thread_exception = None

        self._numThreadsProcessing = 0
        
        self._pollThreads = []
        for i in range(NUM_POLL_THREADS):
            pt = threading.Thread(target=self._queuePoll)
            pt.daemon = False
            pt.start()
            self._pollThreads.append(pt)
        
        self.md = MetaDataHandler.NestedClassMDHandler()
        self.evtLogger = EventLogger(self)
        
        self.sequenceID = genSequenceID()
        self.md['imageID'] = self.sequenceID  
        
        sp.Spooler.__init__(self, filename, frameSource, **kwargs)

        self._lastFrameTime = 1e12

        self.compSettings = {}
        self.compSettings.update(defaultCompSettings)
        try:
            self.compSettings.update(kwargs['compressionSettings'])
        except KeyError:
            pass
            
    def _queuePoll(self):
        while self._dPoll:
            try:
                data = self._postQueue.get_nowait()

                with self._lock:
                    self._numThreadsProcessing += 1

                try:
                    files = []
                    for imNum, frame in data:
                        if self._aggregate_h5:
                            fn = '/'.join(['__aggregate_h5', self.seriesName, 'frame%05d.pzf' % imNum])
                        else:
                            fn = '/'.join([self.seriesName, 'frame%05d.pzf' % imNum])
                            
                        pzf = PZFFormat.dumps(frame, sequenceID=self.sequenceID, frameNum = imNum, **self.compSettings)

                        files.append((fn, pzf))

                    if len(files) > 0:
                        clusterIO.putFiles(files, serverfilter=self.clusterFilter)
                        
                except Exception as e:
                    self._last_thread_exception = e
                    logging.exception('Exception whilst putting files')
                    raise
                finally:
                    with self._lock:
                        self._numThreadsProcessing -= 1

                time.sleep(.01)
                #print 't', len(data)
            except Queue.Empty:
                time.sleep(.01)
                
    def finished(self):
        if not self._last_thread_exception is None:
            #raise an exception here, in the calling thread
            logging.error('An exception occurred in one of the spooling threads')
            raise RuntimeError('An exception occurred in one of the spooling threads')
        else:
            return self._postQueue.empty() and (self._numThreadsProcessing == 0)

        
    def getURL(self):
        #print CLUSTERID, self.seriesName
        return 'PYME-CLUSTER://%s/%s' % (self.clusterFilter, self.seriesName)
        
    def StartSpool(self):
        sp.Spooler.StartSpool(self)
        
        if self._aggregate_h5:
            clusterIO.putFile('__aggregate_h5/' + self.seriesName + '/metadata.json', self.md.to_JSON(), serverfilter=self.clusterFilter)
        else:
            clusterIO.putFile(self.seriesName  + '/metadata.json', self.md.to_JSON(), serverfilter=self.clusterFilter)
    
    def StopSpool(self):
        self._dPoll = False
        sp.Spooler.StopSpool(self)
        
        logger.debug('Stopping spooling %s' % self.seriesName)
        
        if self._aggregate_h5:
            clusterIO.putFile('__aggregate_h5/' + self.seriesName + '/final_metadata.json', self.md.to_JSON(),
                              serverfilter=self.clusterFilter)
    
            #save the acquisition events as json - TODO - consider a binary format as the events
            #can be quite numerous
            clusterIO.putFile('__aggregate_h5/' + self.seriesName + '/events.json', self.evtLogger.to_JSON(),
                              serverfilter=self.clusterFilter)
        
        else:
            clusterIO.putFile(self.seriesName  + '/final_metadata.json', self.md.to_JSON(), serverfilter=self.clusterFilter)
            
            #save the acquisition events as json - TODO - consider a binary format as the events
            #can be quite numerous
            clusterIO.putFile(self.seriesName  + '/events.json', self.evtLogger.to_JSON(), serverfilter=self.clusterFilter)
        
        
    def OnFrame(self, sender, frameData, **kwargs):
        # NOTE: copy is now performed in frameWrangler, so we don't need to worry about it here
        if frameData.shape[0] == 1:
            self._buffer.append((self.imNum, frameData))
        else:
            self._buffer.append((self.imNum, frameData.reshape(1, frameData.shape[0], frameData.shape[1])))

        #print len(self.buffer)
        t = time.time()

        #purge buffer if more than  self.buflen frames have been added, or more than 1 second elapsed
        if (len(self._buffer) >= self.buflen) or ((t - self._lastFrameTime) > 1):
            self.FlushBuffer()
            self._lastFrameTime = t
        
        sp.Spooler.OnFrame(self)
        
    def cleanup(self):
        self._dPoll = False
      
    def FlushBuffer(self):
      self._postQueue.put(self._buffer)
      self._buffer = []
     



   
