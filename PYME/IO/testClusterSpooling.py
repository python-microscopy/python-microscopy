"""
This file provides a minimal test for spooling into the cluster with fake data, and with a multi-threaded spooler

Setup and Usage:
=================
- Run PYMEDataServer instances (on either local or remote host) with a -t option. This activates a testing mode where
  data is dumped on arrival rather than being written to disk.
- Optionally use -p portNum and -v HTTP_version options on server
- Adjust the frame size (below) as appropriate. 2k by 2k should be the standard initial benchmark as at this size per-frame
  overhead should not be excessive. We can optimize for per-frame overhead later. TODO - make this a command line option.
- Run this file

Performance on Macbook to 10 local servers (as of 19/7/2016):
========================================================================
-----------------------------------------------------------------
|                 | HTTP/1.0      | HTTP/1.1 with keep-alive    |
-----------------------------------------------------------------
| 2kx2k frames    | 420 MB/s      | 455 MB/s                    |
| 200x800 frames  | 133 MB/s      | 372 MB/s                    |
-----------------------------------------------------------------

New benchmark on macbook to 10 local servers (25/7/2016) using raw sockets (HTTP/1.1 only)
===========================================================================================

-----------------------------------------------------------------
|                 | HTTP/1.0      | HTTP/1.1 with keep-alive    |
-----------------------------------------------------------------
| 2kx2k frames    | N/A           | 533 MB/s  (69 FPS)          |
| 200x800 frames  | N/A           | 723 MB/s  (2260 FPS)        |
-----------------------------------------------------------------

Target: 800 MB/s
================


Notes
=====
- Occasional connection timeout bug. Seems to be worse with HTTP/1.1
- Now extened to permit testing of direct streaming to HDF (hacky)
"""

import glob
import json
import time
import os
import sys

from six.moves import xrange

from PYME.IO.clusterExport import ImageFrameSource, MDSource
from PYME.IO import MetaDataHandler
from PYME.IO.DataSources import DcimgDataSource, MultiviewDataSource
from PYME.Analysis import MetaData
from PYME.IO import HDFSpooler
from PYME.IO import HTTPSpooler_v2 as HTTPSpooler
from PYME.IO import PZFFormat

import time
from PYME.contrib import dispatch
import sys

TEST_FRAME_SIZE = [2000,2000]
TEST_CHUNK_SIZE = 50

import numpy as np

from PYME.util import fProfile

# class DCIMGSpooler(object):
class TestSpooler:
    def __init__(self, testFrameSize = TEST_FRAME_SIZE, serverfilter=''):
        

        self.testData = (100*np.random.rand(*testFrameSize)).astype('uint16')
        self.onFrame = dispatch.Signal(['frameData'])
        self.spoolProgress = dispatch.Signal(['percent'])

        self.mdh = MetaDataHandler.NestedClassMDHandler()
        self.serverfilter=serverfilter


    def run(self, filename=None, nFrames = 2000, interval=0, compression=False, hdf_spooler=False, frameShape=None):
        if filename is None:
            filename = 'test_%3.1f' % time.time()

        self.imgSource = ImageFrameSource()
        self.metadataSource = MDSource(self.mdh)
        MetaDataHandler.provideStartMetadata.append(self.metadataSource)

        #generate the spooler
        if hdf_spooler:
            if compression:
                self.spooler = HDFSpooler.Spooler(filename, self.onFrame, frameShape = frameShape, complevel=0)
            else:
                self.spooler = HDFSpooler.Spooler(filename, self.onFrame,frameShape = frameShape, complevel=3)
        else:
            if compression:
                self.spooler = HTTPSpooler.Spooler(filename, self.onFrame, frameShape = None, serverfilter=self.serverfilter)
            else:
                self.spooler = HTTPSpooler.Spooler(filename, self.onFrame,
                                                   frameShape = None, serverfilter=self.serverfilter,
                                                   compressionSettings={'compression': PZFFormat.DATA_COMP_RAW,
                                                                        'quantization':PZFFormat.DATA_QUANT_NONE})
        
        try:
            #spool our data
            self.spooler.StartSpool()
    
            print(getattr(self.spooler, 'seriesName', self.spooler.filename))
    
            startTime = time.time()
    
            self._spoolData(nFrames, interval)

            self.spooler.StopSpool()
            #wait until we've sent everything
            #this is a bit of a hack
            time.sleep(.1)
            while not self.spooler.finished():
                time.sleep(.1)
    
            endTime = time.time()
            duration = endTime - startTime
    
            print('######################################')
            print('%d frames spooled in %f seconds' % (nFrames, duration))
            print('%3.0f frames per second' % (nFrames/duration))
            print('Avg throughput: %3.0f MB/s' % (nFrames*self.testData.nbytes/(1e6*duration)))
    
        finally:
            self.spooler.cleanup()



    def _spoolData(self, nFrames, interval=0):
        for i in xrange(nFrames):
            self.onFrame.send(self, frameData=self.testData)
            if (i % 100) == 0:
                self.spoolProgress.send(self, percent=float(i)/nFrames)
                print('Spooling %d of %d frames' % (i, nFrames))

            if interval > 0:
                time.sleep(interval)


#PROFILE = False


if __name__ == '__main__':
    prof = fProfile.thread_profiler()

    from optparse import OptionParser
    op = OptionParser(usage='usage: %s [options] [filename]' % sys.argv[0])

    op.add_option('-n', '--num-frames', dest='nframes', default=2000,
                  help="Number of frames to spool")
    op.add_option('-p', '--profile-output', dest='profile_out', help="Filename to save profiling info. NB this will slow down the server",
                  default=None)
    op.add_option('-s', '--frame-size', dest='frame_size', help="The frame size in an AxB format", default="2000x2000")
    op.add_option('-c', '--compression', dest='compression', help="Use compression", default=False, action='store_true')

    options, args = op.parse_args()

    PROFILE = False

    if not options.profile_out is None:
        prof.profileOn('.*PYME.*|.*requests.*|.*socket.*', options.profile_out)
        PROFILE = True

    frameSize = [int(s) for s in options.frame_size.split('x')]
    ts = TestSpooler(testFrameSize=frameSize)
    ts.run(nFrames=int(options.nframes), compression=options.compression)

    if PROFILE:
        prof.profileOff()






