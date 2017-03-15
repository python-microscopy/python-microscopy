import subprocess
from PYME.IO import clusterIO
import tempfile
import os
import shutil
#import unittest
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
    
    
def test_put():
    testdata = 'foo bar\n'
    clusterIO.putFile('test.txt', testdata, 'TEST')
    retrieved = clusterIO.getFile('test.txt', 'TEST')
    
    assert testdata == retrieved
    
def test_putfiles_and_list():
    test_files = [('test_list/file_%d' % i, 'testing ... \n') for i in range(10)]
    
    clusterIO.putFiles(test_files, 'TEST')
    
    listing = clusterIO.listdir('test_list/')
    
    assert(len(listing) == 10)


def test_double_put():
    """Trying to put the same file twice should cause an error"""
    testdata = 'foo bar\n'

    clusterIO.putFile('test_d.txt', testdata, 'TEST')
    
    try:
        clusterIO.putFile('test_d.txt', testdata, 'TEST')
        raise AssertionError('Second put attempt did not raise an error')
    except RuntimeError:
        #we want to generate this error
        pass
    
    #retrieved = clusterIO.getFile('test.txt', 'TEST')
    
    #assert testdata == retrieved