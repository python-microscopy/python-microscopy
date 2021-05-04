import subprocess
from PYME.IO import clusterIO
import tempfile
import os
import shutil
import signal
import pytest
import time
import sys

import logging
logger = logging.getLogger(__name__)
proc = None
tmp_root = None

def setup_module():
    global proc, tmp_root
    tmp_root = os.path.join(tempfile.gettempdir(), 'PYMEDataServer_TES1')
    
    print('DataServer root: %s' % tmp_root)
    
    if os.path.exists(tmp_root):
        print('Removing existing temp spooler dir')
        shutil.rmtree(tmp_root)
        
    os.makedirs(tmp_root)
    proc = subprocess.Popen([sys.executable, '-m', 'PYME.cluster.HTTPDataServer',  '-r', tmp_root, '-f', 'TES1', '-a', 'local'])
    
    time.sleep(5) #give time for the server to spin up
    
    logging.info('Advertised services:\n------------------\n%s' % '\n'.join([str(s) for s in clusterIO.get_ns().get_advertised_services()]))
    
    
def teardown_module():
    global proc, tmp_root
    proc.send_signal(signal.SIGINT)
    time.sleep(5)
    proc.kill()
    
    #shutil.rmtree(tmp_root)
    
    
def test_put():
    testdata = b'foo bar\n'
    clusterIO.put_file('_testing/test.txt', testdata, 'TES1')
    retrieved = clusterIO.get_file('_testing/test.txt', 'TES1')
    
    assert testdata == retrieved
    
def test_putfiles_and_list():
    test_files = [('_testing/test_list/file_%d' % i, b'testing ... \n') for i in range(10)]
    
    clusterIO.put_files(test_files, 'TES1')
    
    listing = clusterIO.listdir('_testing/test_list/', 'TES1')
    
    assert(len(listing) == 10)


def test_list_after_timeout():
    test_files = [('_testing/test_list2/file_%d' % i, b'testing ... \n') for i in range(10)]
    
    clusterIO.put_files(test_files, 'TES1')
    
    time.sleep(2)
    listing = clusterIO.listdirectory('_testing/test_list2/', 'TES1',timeout=.00001)
    listing = clusterIO.listdirectory('_testing/test_list2/', 'TES1', timeout=5)
    
    assert (len(listing) == 10)


def test_double_put():
    """Trying to put the same file twice should cause an error"""
    testdata = b'foo bar\n'

    clusterIO.put_file('_testing/test_d.txt', testdata, 'TES1')
    
    try:
        clusterIO.put_file('_testing/test_d.txt', testdata, 'TES1')
        raise AssertionError('Second put attempt did not raise an error')
    except RuntimeError:
        #we want to generate this error
        pass
    
    #retrieved = clusterIO.getFile('test.txt', 'TES1')
    
    #assert testdata == retrieved
    
def test_aggregate_h5r():
    import numpy as np
    from PYME.IO import clusterResults
    testdata = np.ones(10, dtype=[('a', '<f4'), ('b', '<f4')])
    
    clusterResults.fileResults('pyme-cluster://TES1/__aggregate_h5r/_testing/test_results.h5r/foo', testdata)
    clusterResults.fileResults('pyme-cluster://TES1/__aggregate_h5r/_testing/test_results.h5r/foo', testdata)
    clusterResults.fileResults('pyme-cluster://TES1/__aggregate_h5r/_testing/test_results.h5r/foo', testdata)


def test_dircache_purge():
    testdata = b'foo bar\n'
    for i in range(1050):
        clusterIO.put_file('_testing/lots_of_folders/test_%d/test.txt' % i, testdata, 'TES1')
    
        listing = clusterIO.listdir('_testing/lots_of_folders/test_%d/' % i, 'TES1')
    
    #assert (len(listing) == 10)

@pytest.mark.skip(reason='Test is malformed and has race condition (results file should be created before multi threaded access)')
def test_mulithread_result_filing():
    # FIXME - this test is expected to fail as files should be created before multi-threaded aggregate operations
    # with enough of a delay between creation and access to ensure that the file is present in directory caches.
    import numpy as np
    from PYME.IO import clusterResults, unifiedIO
    import tables
    import posixpath
    import threading
    
    n_filings = 500
    n_per = np.random.randint(0, 100, n_filings)
    data = [np.ones(n_per[ind], dtype=[('a', '<f4'), ('b', '<f4')]) for ind in range(n_filings)]
    dest = 'pyme-cluster://TES1/__aggregate_h5r/_testing/test_result_filing.h5r'

    threads = []
    for ind in range(n_filings):
        t = threading.Thread(target=clusterResults.fileResults, 
                             args=(posixpath.join(dest, 'foo'), data[ind]))
        t.start()
        threads.append(t)
    
    [t.join() for t in threads]

    time.sleep(5)

    with unifiedIO.local_or_temp_filename('pyme-cluster://TES1/_testing/test_result_filing.h5r') as f,\
        tables.open_file(f) as t:
        n_received = len(t.root.foo)
    
    assert n_received == np.sum(n_per)
