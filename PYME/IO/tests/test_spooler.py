import subprocess
from PYME.IO import clusterIO
from PYME.Acquire import HTTPSpooler
from PYME.IO import testClusterSpooling
import tempfile
import os
import shutil

from PYME.IO.clusterExport import ImageFrameSource, MDSource
from PYME.IO import MetaDataHandler
#import unittest
import time

procs = []
tmp_root = None

def setup_module():
    global proc, tmp_root
    tmp_root = os.path.join(tempfile.gettempdir(), 'PYMEDataServer_TEST')
    os.makedirs(tmp_root)
    port_start = 8100
    for i in range(10):
        proc = subprocess.Popen('PYMEDataServer -r %s -f TEST -t -p %d' % (tmp_root, port_start + i), shell=True)
        procs.append(proc)
        
    time.sleep(5)


def teardown_module():
    global proc, tmp_root
    #proc.send_signal(1)
    #time.sleep(1)
    for proc in procs:
        proc.kill()
    
    shutil.rmtree(tmp_root)
    
    
def test_spooler():
    ts = testClusterSpooling.TestSpooler(testFrameSize=[256,256], serverfilter='TEST')
    ts.run(nFrames=600)
    
    
if __name__ == '__main__':
    setup_module()
    test_spooler()
    teardown_module()