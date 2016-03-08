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

import threading
import Queue

from PYME.ParallelTasks import clusterIO
from PYME.FileUtils import PZFFormat

import numpy as np
import random

import json

class EventLogger:
    def __init__(self, spool, scope):
        self.spooler = spool
        self.scope = scope
          
        self._events = []
    
    def logEvent(self, eventName, eventDescr = ''):
        if eventName == 'StartAq':
            eventDescr = '%d' % self.spooler.imNum
        self._events.append((eventName, eventDescr, sp.timeFcn()))
        
    def to_JSON(self):
        return json.dumps(self._events)
          

CLUSTERID=''  

def genSequenceID(filename=''):
    return  int(time.time()) & random.randint(0, 2**31) << 31 

#Push data to cluster from multiple threads simultaeneously to hide IO latency
#of each individual node. Not sure what the best number is here - currenty set
#a "safe" maximum number of nodes that could access data
NUM_POLL_THREADS = 1

class Spooler(sp.Spooler):
    def __init__(self, scope, filename, acquisator, protocol = p.NullProtocol, parent=None, complevel=2, complib='zlib'):
        
        filename = filename[len(nameUtils.datadir):]
        
        self.seriesName = '/'.join(filename.split(os.path.sep))
        if self.seriesName.startswith('/'):
            self.seriesName = self.seriesName[1:]
        print filename, self.seriesName
        self.buffer = []
        
        self.buflen = 50
        
        self.postQueue = Queue.Queue()
        self.dPoll = True
        
        self.pollThreads = []
        for i in range(NUM_POLL_THREADS):
            pt = threading.Thread(target=self._queuePoll)
            pt.start()
            self.pollThreads.append(pt)
        
        self.md = MetaDataHandler.NestedClassMDHandler()
        self.evtLogger = EventLogger(self, scope)
        
        self.sequenceID = genSequenceID()
        self.md['imageID'] = self.sequenceID  
        
        sp.Spooler.__init__(self, scope, filename, acquisator, protocol, parent)
        
            
    def _queuePoll(self):
        while self.dPoll:            
            data = self.postQueue.get()
            
            files = []
            for imNum, frame in data:
                fn = '/'.join([self.seriesName, 'frame%05d.pzf' % imNum])
                pzf = PZFFormat.dumps(frame, sequenceID=self.sequenceID, frameNum = imNum, compression='huffman')
                
                files.append((fn, pzf))
                
            if len(files) > 0:
                clusterIO.putFiles(files)
                
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
        
        
    def Tick(self, caller): 
        self.buffer.append((self.imNum, caller.dsa.copy()))

        if len(self.buffer) >= self.buflen:
            self.FlushBuffer()
        
        sp.Spooler.Tick(self, caller)
      
    def FlushBuffer(self):
      self.postQueue.put(self.buffer)
      self.buffer = []
     



   
