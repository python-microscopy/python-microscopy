"""
This module attempts to replicate the race-conditions and discovery problems we are seeing when trying to configure large
numbers of data servers and workers using zeroconf.
"""
import subprocess
#from PYME.IO import clusterIO
import tempfile
import os
import shutil
import multiprocessing
import time
import random

#logger = logging.getLogger(__name__)

server_procs = []
tmp_root = None

def setup_module():
    global tmp_root
    tmp_root = os.path.join(tempfile.gettempdir(), 'PYMEDataServer_TEST')
    for i in range(10):
        srv_root = os.path.join(tmp_root, 'srv%d' % i)
        os.makedirs(srv_root)
        print('Launching server %d' % i)
        proc = subprocess.Popen('python -m PYME.cluster.HTTPDataServer  -r %s -f TEST -p 808%d' % (srv_root, i) , shell=True)
        server_procs.append(proc)
        
    time.sleep(5)
        
        
def _discover_servers(foo=None):
    #import logging
    #logging.basicConfig(filename='test_discovery_%s.log' % foo, level=logging.DEBUG)
    #logging.debug('About to list servers')
    #from PYME.misc import pyme_zeroconf as pzc
    from PYME.IO import clusterIO
    
    #time.sleep(10*random.random())
    
    print('discovering %s' % foo)
    #time.sleep(3 * random.random())
    #ns = pzc.getNS('_pyme-http')
    #time.sleep(5)
    
    ns = clusterIO.get_ns()
    
    return ns.list('TEST')
    
def _test_discovery():
    pool = multiprocessing.Pool(60)
    
    detections = pool.map(_discover_servers, range(60))
    
    #print detections
    
    for d in detections:
        #print d
        print(len(d))
        #assert(len(d) == 10)
        
    #assert(False)


def teardown_module():
    global tmp_root

    for proc in server_procs:
        proc.kill()
    
    shutil.rmtree(tmp_root)
    
    
if __name__ == '__main__':
    setup_module()
    
    try:
        _test_discovery()
    finally:
        teardown_module()