import subprocess
from PYME.IO import clusterIO
import tempfile
import os
import shutil
#import unittest
import time
import sys

import logging
logger = logging.getLogger(__name__)
proc = None
tmp_root = None

def setup_module():
    global proc, tmp_root
    tmp_root = os.path.join(tempfile.gettempdir(), 'PYMEDataServer_TEST')
    
    print('DataServer root: %s' % tmp_root)
    
    if os.path.exists(tmp_root):
        print('Removing existing temp spooler dir')
        shutil.rmtree(tmp_root)
        
    os.makedirs(tmp_root)
    proc = subprocess.Popen([sys.executable, '-m', 'PYME.cluster.HTTPDataServer',  '-r', tmp_root, '-f', 'TEST', '-a', 'local'])
    
    time.sleep(3) #give time for the server to spin up
    
    logging.info('Advertised services:\n------------------\n%s' % '\n'.join([str(s) for s in clusterIO.get_ns().get_advertised_services()]))
    
    
def teardown_module():
    global proc, tmp_root
    #proc.send_signal(1)
    #time.sleep(1)
    proc.kill()
    
    #shutil.rmtree(tmp_root)
    
    
def test_put():
    testdata = b'foo bar\n'
    clusterIO.put_file('_testing/test.txt', testdata, 'TEST')
    retrieved = clusterIO.get_file('_testing/test.txt', 'TEST')
    
    assert testdata == retrieved
    
def test_putfiles_and_list():
    test_files = [('_testing/test_list/file_%d' % i, b'testing ... \n') for i in range(10)]
    
    clusterIO.put_files(test_files, 'TEST')
    
    listing = clusterIO.listdir('_testing/test_list/', 'TEST')
    
    assert(len(listing) == 10)


def test_list_after_timeout():
    test_files = [('_testing/test_list2/file_%d' % i, b'testing ... \n') for i in range(10)]
    
    clusterIO.put_files(test_files, 'TEST')
    
    time.sleep(2)
    listing = clusterIO.listdirectory('_testing/test_list2/', 'TEST',timeout=.00001)
    listing = clusterIO.listdirectory('_testing/test_list2/', 'TEST', timeout=5)
    
    assert (len(listing) == 10)


def test_double_put():
    """Trying to put the same file twice should cause an error"""
    testdata = b'foo bar\n'

    clusterIO.put_file('_testing/test_d.txt', testdata, 'TEST')
    
    try:
        clusterIO.put_file('_testing/test_d.txt', testdata, 'TEST')
        raise AssertionError('Second put attempt did not raise an error')
    except RuntimeError:
        #we want to generate this error
        pass
    
    #retrieved = clusterIO.getFile('test.txt', 'TEST')
    
    #assert testdata == retrieved
    
def test_aggregate_h5r():
    import numpy as np
    from PYME.IO import clusterResults
    testdata = np.ones(10, dtype=[('a', '<f4'), ('b', '<f4')])
    
    clusterResults.fileResults('pyme-cluster://TEST/__aggregate_h5r/_testing/test_results.h5r/foo', testdata)
    clusterResults.fileResults('pyme-cluster://TEST/__aggregate_h5r/_testing/test_results.h5r/foo', testdata)
    clusterResults.fileResults('pyme-cluster://TEST/__aggregate_h5r/_testing/test_results.h5r/foo', testdata)


def test_dircache_purge():
    testdata = b'foo bar\n'
    for i in range(1050):
        clusterIO.put_file('_testing/lots_of_folders/test_%d/test.txt' % i, testdata, 'TEST')
    
        listing = clusterIO.listdir('_testing/lots_of_folders/test_%d/' % i, 'TEST')
    
    #assert (len(listing) == 10)