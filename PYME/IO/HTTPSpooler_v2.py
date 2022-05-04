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

#from PYME.IO import MetaDataHandler
from PYME.IO.events import MemoryEventLogger
import PYME.IO.Spooler as sp

import os
# import time
# import threading

# try:
#     # noinspection PyCompatibility
#     import Queue # type: ignore
# except ImportError:
#     #py3
#     import queue as Queue

from PYME.IO import clusterIO
#from PYME.IO import cluster_streaming
#from PYME.IO import PZFFormat
from PYME import config

#import random
import logging
logger = logging.getLogger(__name__)


def getReducedFilename(filename):
    #rname = filename[len(nameUtils.datadir):]
        
    sname = '/'.join(filename.split(os.path.sep))
    if sname.startswith('/'):
        sname = sname[1:]
    
    return sname
    
def exists(seriesName):
    return clusterIO.exists(getReducedFilename(seriesName))


from PYME.IO import acquisition_backends

class Spooler(sp.Spooler):
    def __init__(self, filename, frameSource, frameShape, **kwargs):
        sp.Spooler.__init__(self, filename, frameSource, **kwargs)
        
        self.seriesName = getReducedFilename(filename)        
        self._aggregate_h5 = kwargs.get('aggregate_h5', False)
        
        self.clusterFilter = kwargs.get('serverfilter', 
                                        config.get('dataserver-filter', ''))
        
        
        chunk_size = config.get('httpspooler-chunksize', 50)
        def dist_fcn(n_servers, i=None):
            if i is None:
                # distribute at random
                import random
                return random.randrange(n_servers)
            
            return int(i/chunk_size) % n_servers
        
        
        self._backend = acquisition_backends.ClusterBackend(self.seriesName, 
                                                            distribution_fcn=dist_fcn, 
                                                            compression_settings=kwargs.get('compressionSettings', {}),
                                                            cluster_h5=self._aggregate_h5,
                                                            serverfilter=self.clusterFilter,
                                                            shape=[-1,-1,1,-1,1], #spooled aquisitions are time series (for now)
                                                            )
        
        self.md = self._backend.mdh
        self.evtLogger = MemoryEventLogger(self, time_fcn=self._time_fcn)
        self._stopping = False
                
    def finished(self):
        # FIXME - this probably needs a bit more work.
        return self._stopping
        
    def getURL(self):
        #print CLUSTERID, self.seriesName
        return 'PYME-CLUSTER://%s/%s' % (self.clusterFilter, self.seriesName)
        
    def StartSpool(self):
        sp.Spooler.StartSpool(self)
        
        logger.debug('Starting spooling: %s' %self.seriesName)

        self._backend.initialise()
    
    def finalise(self):
        # wait until our input queue is empty rather than immediately stopping saving.
        self._stopping=True
        logger.debug('Stopping spooling %s' % self.seriesName)
        
        #TODO - add option for non-blocking finalise??
        self._backend.finalise(events=None)
        
        # save events
        # do this here, **after** we have joined all the pushing threads to make sure that writing to all nodes has
        # finished when we save events.
        # TODO - use a binary format for saving events - they can be quite
        # numerous, and can trip the standard 1 s clusterIO.put_file timeout.
        clusterIO.put_file(self._backend._series_location + '/events.json', 
                               self.evtLogger.to_JSON().encode(),
                               self.clusterFilter, timeout=10)
        

    def OnFrame(self, sender, frameData, **kwargs):
        self._backend.store_frame(self.imNum, frameData)

        sp.Spooler.OnFrame(self)
        
    def cleanup(self):
        del self._backend
     



   
