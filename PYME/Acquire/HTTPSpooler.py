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
from PYME import config

import json

import logging
logger = logging.getLogger(__name__)

class EventLogger:
    def __init__(self, spool):#, scope):
        self.spooler = spool
        #self.scope = scope
          
        self._events = []
        self._event_lock = threading.Lock()
    
    def logEvent(self, eventName, eventDescr = '', timestamp=None):
        if eventName == 'StartAq' and eventDescr == '':
            eventDescr = '%d' % self.spooler.imNum

        if timestamp is None:
            timestamp = sp.timeFcn()
            
        with self._event_lock:
            self._events.append((eventName, eventDescr, timestamp))
        
    def to_JSON(self):
        return json.dumps(self._events)
          

def genSequenceID(filename=''):
    return  int(time.time()) & random.randint(0, 2**31) << 31 
    

    

def getReducedFilename(filename):
    #rname = filename[len(nameUtils.datadir):]
        
    sname = '/'.join(filename.split(os.path.sep))
    if sname.startswith('/'):
        sname = sname[1:]
    
    return sname
    
def exists(seriesName):
    return clusterIO.exists(getReducedFilename(seriesName))

#Push data to cluster from multiple threads simultaeneously to hide IO latency
#of each individual node. Not sure what the best number is here - currenty set
#a "safe" maximum number of nodes that could access data
NUM_POLL_THREADS = 10
QUEUE_MAX_SIZE = 200 # ~10k frames

defaultCompSettings = {
    'compression' : PZFFormat.DATA_COMP_HUFFCODE,
    'quantization' : PZFFormat.DATA_QUANT_NONE,
    'quantizationOffset' : -1e6, # set to an unreasonable value so that we raise an error if default offset is used
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
        
        self.clusterFilter = kwargs.get('serverfilter', 
                                        config.get('dataserver-filter', ''))
        self._buffer = []
        
        self.buflen = config.get('httpspooler-chunksize', 50)
        
        self._postQueue = Queue.Queue(QUEUE_MAX_SIZE)
        self._dPoll = True
        self._stopping = False
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
        
        if not self.compSettings['quantization'] == PZFFormat.DATA_QUANT_NONE:
            # do some sanity checks on our quantization parameters
            # note that these conversions will throw a ValueError if the settings are not numeric
            offset = float(self.compSettings['quantizationOffset'])
            scale = float(self.compSettings['quantizationScale'])
            
            # these are potentially a bit too permissive, but should catch an offset which has been left at the
            # default value
            assert(offset >= 0)
            assert(scale >=.001)
            assert(scale <= 100)
            
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
                        clusterIO.put_files(files, serverfilter=self.clusterFilter)
                        
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
                if self._stopping:
                    self._dPoll = False
                else:
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
        
        logger.debug('Starting spooling: %s' %self.seriesName)
        
        if self._aggregate_h5:
            #NOTE: allow a longer timeout than normal here as __aggregate with metadata waits for a lock on the server side before
            # actually adding (and is therefore susceptible to longer latencies than most operations). FIXME - remove server side lock.
            clusterIO.put_file('__aggregate_h5/' + self.seriesName + '/metadata.json', self.md.to_JSON().encode(), serverfilter=self.clusterFilter, timeout=3)
        else:
            clusterIO.put_file(self.seriesName + '/metadata.json', self.md.to_JSON().encode(), serverfilter=self.clusterFilter)
    
    def StopSpool(self):
        sp.Spooler.StopSpool(self)

        # wait until our input queue is empty rather than immediately stopping saving.
        self._stopping=True
        logger.debug('Stopping spooling %s' % self.seriesName)
        
        
        #join our polling threads
        if config.get('httpspooler-jointhreads', True):
            # Allow this to be switched off in a config option for maximum performance on High Throughput system.
            # Joining threads is the recommended and safest behaviour, but forces spooling of current series to complete
            # before next series starts, so could have negative performance implications.
            # The alternative - letting spooling continue during the acquisition of the next series - has the potential
            # to result in runaway memory and thread usage when things go pear shaped (i.e. spooling is not fast enough)
            # TODO - is there actually a performance impact that justifies this config option, or is it purely theoretical
            for pt in self._pollThreads:
                pt.join()
        
        if self._aggregate_h5:
            clusterIO.put_file('__aggregate_h5/' + self.seriesName + '/final_metadata.json', self.md.to_JSON().encode(),
                               serverfilter=self.clusterFilter)
    
            #save the acquisition events as json - TODO - consider a binary format as the events
            #can be quite numerous
            clusterIO.put_file('__aggregate_h5/' + self.seriesName + '/events.json', self.evtLogger.to_JSON().encode(),
                               serverfilter=self.clusterFilter)
        
        else:
            clusterIO.put_file(self.seriesName + '/final_metadata.json', self.md.to_JSON().encode(), serverfilter=self.clusterFilter)
            
            #save the acquisition events as json - TODO - consider a binary format as the events
            #can be quite numerous
            clusterIO.put_file(self.seriesName + '/events.json', self.evtLogger.to_JSON().encode(), serverfilter=self.clusterFilter)
        
        
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
     



   
