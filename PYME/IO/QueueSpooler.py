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
import os
import time

import PYME.IO.Spooler as sp
#from PYME.Acquire import protocol as p
from PYME.IO.FileUtils import fileID

try:
    from PYME.misc import hybrid_ns
    
    ns = hybrid_ns.getNS()
except ImportError:
    ns = None
      

class Spooler(sp.Spooler):
    def __init__(self, filename, frameSource, frameShape, complevel=6, complib='zlib', **kwargs):
#       if 'PYME_TASKQUEUENAME' in os.environ.keys():
#            taskQueueName = os.environ['PYME_TASKQUEUENAME']
#       else:
#            taskQueueName = 'taskQueue'
        import Pyro.core
        from PYME.IO.events import QueueEventLogger
        sp.Spooler.__init__(self, filename, frameSource, **kwargs)

        from PYME.misc.computerName import GetComputerName
        compName = GetComputerName()

        taskQueueName = 'TaskQueues.%s' % compName

        if ns:
            URI = ns.resolve(taskQueueName)
        else:
            URI = 'PYRONAME://' + taskQueueName

        self.tq = Pyro.core.getProxyForURI(URI)
        self.tq._setOneway(['postTask', 'postTasks', 'addQueueEvents', 'setQueueMetaData', 'logQueueEvent'])

        self.seriesName = filename
        self.buffer = []
        self.buflen = 30

        self.tq.createQueue('HDFTaskQueue',self.seriesName, filename, frameSize = frameShape, complevel=complevel, complib=complib)

        self.md = MetaDataHandler.QueueMDHandler(self.tq, self.seriesName)
        self.evtLogger = QueueEventLogger(self, self.tq, self.seriesName, time_fcn=self._time_fcn)
   
    def OnFrame(self, sender, frameData, **kwargs):
        if not self.watchingFrames:
            #we have already disconnected
            return
          
        #self.tq.postTask(cSMI.CDataStack_AsArray(caller.ds, 0).reshape(1,self.scope.cam.GetPicWidth(),self.scope.cam.GetPicHeight()), self.seriesName)

        # NOTE: copy is now performed in frameWrangler, so we don't need to worry about it here
        if frameData.shape[0] == 1:
            self.buffer.append(frameData)
        else:
            self.buffer.append(frameData.reshape(1,frameData.shape[0],frameData.shape[1]))

        if self.imNum == 0: #first frame
            self.md.setEntry('imageID', fileID.genFrameID(self.buffer[-1].squeeze()))

        if (len(self.buffer) >= self.buflen):
            self.FlushBuffer()

        sp.Spooler.OnFrame(self)
      
    def FlushBuffer(self):
        t1 = time.time()
        self.tq.postTasks(self.buffer, self.seriesName)
        #print time.time() -t1
        self.buffer = []



   
