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
import Queue

from PYME.IO import clusterIO
from PYME.IO import PZFFormat

import numpy as np
import random

import json

class EventLogger:
    def __init__(self, spool):#, scope):
        self.spooler = spool
        #self.scope = scope
          
        self._events = []
    
    def logEvent(self, eventName, eventDescr = '', timestamp=None):
        if eventName == 'StartAq':
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
        self.buffer = []
        
        self.buflen = 50
        
        self.postQueue = Queue.Queue()
        self.dPoll = True
        self._lock = threading.Lock()

        self.numThreadsProcessing = 0
        
        self.pollThreads = []
        for i in range(NUM_POLL_THREADS):
            pt = threading.Thread(target=self._queuePoll)
            pt.start()
            self.pollThreads.append(pt)
        
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
        while self.dPoll:
            try:
                data = self.postQueue.get_nowait()

                with self._lock:
                    self.numThreadsProcessing += 1

                try:
                    files = []
                    for imNum, frame in data:
                        fn = '/'.join([self.seriesName, 'frame%05d.pzf' % imNum])
                        pzf = PZFFormat.dumps(frame, sequenceID=self.sequenceID, frameNum = imNum, **self.compSettings)

                        files.append((fn, pzf))

                    if len(files) > 0:
                        clusterIO.putFiles(files)

                finally:
                    with self._lock:
                        self.numThreadsProcessing -= 1

                time.sleep(.01)
                #print 't', len(data)
            except Queue.Empty:
                time.sleep(.01)

        
    def getURL(self):
        #print CLUSTERID, self.seriesName
        return 'PYME-CLUSTER://%s/%s' % (CLUSTERID, self.seriesName)
        
    def StartSpool(self):
        sp.Spooler.StartSpool(self)
        clusterIO.putFile(self.seriesName  + '/metadata.json', self.md.to_JSON())
    
    def StopSpool(self):
        self.dPoll = False
        sp.Spooler.StopSpool(self)
        
        clusterIO.putFile(self.seriesName  + '/final_metadata.json', self.md.to_JSON())
        
        #save the acquisition events as json - TODO - consider a binary format as the events
        #can be quite numerous
        clusterIO.putFile(self.seriesName  + '/events.json', self.evtLogger.to_JSON())
        
        
    def OnFrame(self, sender, frameData, **kwargs):
        # NOTE: copy is now performed in frameWrangler, so we don't need to worry about it here
        self.buffer.append((self.imNum, frameData.reshape(1,frameData.shape[0],frameData.shape[1])))

        #print len(self.buffer)
        t = time.time()

        #purge buffer if more than  self.buflen frames have been added, or more than 1 second elapsed
        if (len(self.buffer) >= self.buflen) or ((t - self._lastFrameTime) > 1):
            self.FlushBuffer()
            self._lastFrameTime = t
        
        sp.Spooler.OnFrame(self)
      
    def FlushBuffer(self):
      self.postQueue.put(self.buffer)
      self.buffer = []
     



   
