import subprocess
from PYME.IO import testClusterSpooling
import tempfile
import os
import shutil

import time
import sys
import logging
from PYME.IO import clusterIO


procs = []
tmp_root = None

def setup_module():
    global proc, tmp_root
    tmp_root = os.path.join(tempfile.gettempdir(), 'PYMEDataServer_TEST')
    if not os.path.exists(tmp_root):
        os.makedirs(tmp_root)
        
    port_start = 8100
    for i in range(10):
        proc = subprocess.Popen([sys.executable, '-m', 'PYME.cluster.HTTPDataServer', '-r', tmp_root,  '-f', 'TEST', '-t', '-p', '%d' % (port_start + i), '--timeout-test=0.5', '-a', 'local'], stderr= sys.stderr, shell=False)
        procs.append(proc)
        
    time.sleep(5)
    print('Launched servers')
    print('Advertised services:\n------------------\n%s' % '\n'.join([str(s) for s in clusterIO.get_ns().get_advertised_services()]))


def teardown_module():
    global proc, tmp_root
    #proc.send_signal(1)
    #time.sleep(1)
    
    for proc in procs:
        proc.kill()
        
    print('Killed all servers')
    def _wait_for_poll_exit(func, path, _):
        # wait for the server process to exit and release the file lock before we try to delete the spooler dir
        import time
        time.sleep(1)
    shutil.rmtree(tmp_root, onerror=_wait_for_poll_exit)
    
    

def test_spooler(nFrames=50):
    ts = testClusterSpooling.TestSpooler(testFrameSize=[1024,256], serverfilter='TEST')
    ts.run(nFrames=nFrames)
    

from PYME.util import fProfile
    
if __name__ == '__main__':
    prof = fProfile.thread_profiler()
    setup_module()
    try:
        PROFILE = False
    
        if False:
            prof.profileOn('.*PYME.*|.*requests.*|.*socket.*|.*httplib.*', '/Users/david/spool_prof.txt')
            PROFILE = True
            
        test_spooler(500)

        if PROFILE:
            prof.profileOff()
        time.sleep(5)
    finally:
        teardown_module()