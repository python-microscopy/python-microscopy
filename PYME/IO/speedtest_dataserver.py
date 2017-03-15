import subprocess
from PYME.IO import clusterIO
import tempfile
import os
import shutil
#import unittest
import logging
logging.basicConfig(level=logging.DEBUG)
import time

proc = None
tmp_root = None

def setup_module():
    global proc, tmp_root
    tmp_root = os.path.join(tempfile.gettempdir(), 'PYMEDataServer_TEST')
    os.makedirs(tmp_root)
    proc = subprocess.Popen('PYMEDataServer  -r %s -f TEST' % tmp_root , shell=True)
    
    
def teardown_module():
    global proc, tmp_root
    #proc.send_signal(1)
    #time.sleep(1)
    proc.kill()
    
    shutil.rmtree(tmp_root)
    
    
def test_single_put():
    testdata = 'foo bar\n'
    t = time.time()
    clusterIO.putFile('_testing/test.txt', testdata, 'TEST')
    
    print('putting a small file took %3.5f s' % (time.time() - t))

    t = time.time()
    clusterIO.putFile('_testing/test1.txt', testdata, 'TEST')

    print('putting a second small file took %3.5f s' % (time.time() - t))

    t = time.time()
    retrieved = clusterIO.getFile('_testing/test.txt', 'TEST')

    print('retrieving a small file took %3.5f s' % (time.time() - t))
    
    
def test_putfiles_and_list():
    test_files = [('_testing/test_list/file_%d' % i, 'testing ... \n') for i in range(10000)]
    
    t = time.time()
    clusterIO.putFiles(test_files[:1000], 'TEST')
    print('putting 1000 small files took %3.5f s' % (time.time() - t))
    
    t = time.time()
    listing = clusterIO.listdir('_testing/test_list/')
    print('Listing a directory with 1000 small files took %3.5f s' % (time.time() - t))
    
    t = time.time()
    listing = clusterIO.listdir('_testing/test_list/')
    print('Listing a directory with 1000 small files (from local cache) took %3.5f s' % (
    time.time() - t))

    t = time.sleep(5)
    t = time.time()
    listing = clusterIO.listdir('_testing/test_list/')
    print('Listing a directory with 1000 small files (second attempt after local cache expiration) took %3.5f s' % (time.time() - t))
    assert (len(listing) == 1000)
    
    #put remainder
    t = time.time()
    clusterIO.putFiles(test_files[1000:], 'TEST')
    print('putting 9000 small files took %3.5f s' % (time.time() - t))

    time.sleep(2) #should be enough to invalidate local cache

    t = time.time()
    listing = clusterIO.listdir('_testing/test_list/')
    print('Listing a directory with 10000 small files took %3.5f s' % (time.time() - t))

    assert (len(listing) == 10000)

    t = time.time()
    listing = clusterIO.listdir('_testing/test_list/')
    print('Listing a directory with 10000 small files (from local cache) took %3.5f s' % (
        time.time() - t))

    assert (len(listing) == 10000)

    t = time.sleep(1)
    t = time.time()
    listing = clusterIO.listdir('_testing/test_list/')
    print('Listing a directory with 10000 small files (second attempt after local cache expiration) took %3.5f s' % (
    time.time() - t))
    
    assert(len(listing) == 10000)



def run_tests():
    setup_module()
    
    try:
        test_single_put()
        test_putfiles_and_list()
    finally:
        teardown_module()
        
        
if __name__ == '__main__':
    run_tests()