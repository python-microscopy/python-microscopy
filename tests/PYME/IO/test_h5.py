'''
Tests for h5file and h5rfile. With the aim of letting us loosen up pytables pinning / avoiding future pytables issues
(see issue #215)
'''
from PYME.IO import image
from PYME.IO import tabular
from PYME.IO import MetaDataHandler
from PYME.Analysis import MetaData
import numpy as np
import tempfile
import shutil
import os

def test_h5_export_uint16():
    '''
    Saves and re-loads an image using the hdf exporter

    '''
    from PYME.IO import dataExporter
    from PYME.IO import events
    
    data = (1e3*np.random.rand(100,100,50)).astype('uint16')
    evts = np.zeros(3, dtype=events.EVENTS_DTYPE)
    
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_h5.h5')
    
    try:
        dataExporter.ExportData(data, mdh = MetaData.TIRFDefault, events=evts, filename=filename)
    
        im = image.ImageStack(filename=filename)
        
        assert(np.allclose(im.data[:,:,:,0,0].squeeze(), data))
        im.dataSource.release()
    finally:
        shutil.rmtree(tempdir)


def test_h5_export_uint16_multicolour():
    '''
    Saves and re-loads an image using the hdf exporter

    '''
    from PYME.IO import dataExporter
    from PYME.IO import events
    
    data = (1e3 * np.random.rand(100, 100, 50, 2)).astype('uint16')
    evts = np.zeros(3, dtype=events.EVENTS_DTYPE)
    
    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_h5.h5')
    
    try:
        dataExporter.ExportData(data, mdh=MetaData.TIRFDefault, events=evts, filename=filename)
        
        im = image.ImageStack(filename=filename)
        
        assert (np.allclose(im.data[:, :, :, :,:].squeeze(), data))
        im.dataSource.release()
    finally:
        shutil.rmtree(tempdir)
    
 
def test_hdf_spooler(nFrames=50):
    from PYME.IO import testClusterSpooling

    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_spool.h5')
    
    try:
        ts = testClusterSpooling.TestSpooler(testFrameSize=[512,512], serverfilter='TEST')
        ts.run(nFrames=nFrames, filename=filename, hdf_spooler=True, frameShape=[512, 512])

        im = image.ImageStack(filename=filename)
        
        #check image dimensions are as expected
        assert(np.allclose(im.data.shape[:3], [512, 512, 50]))
        
        im.dataSource.release()
    finally:
        shutil.rmtree(tempdir)
        
def test_h5_pzf_put_file():
    '''
    Writes PZF frames via H5File.put_file (the path used by the dataserver for .h5 spooling) and checks that the
    PZFImageIndex is created and the data can be re-loaded. Regression test for numpy 2 `int()` failure on the frame
    number which killed the writer thread after the first frame.
    '''
    from PYME.IO import h5File, PZFFormat
    from PYME.IO.DataSources import HDFDataSource

    frame_nums = [0, 4, 9]
    frames = [(1e3 * np.random.rand(10, 12)).astype('uint16') for _ in frame_nums]

    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_pzf.h5')

    try:
        f = h5File.H5File(filename, 'a')
        with f:
            f.put_file('metadata.json', MetaData.TIRFDefault.to_JSON())
            for fn, d in zip(frame_nums, frames):
                f.put_file('frame%05d.pzf' % fn, PZFFormat.dumps(d, frameNum=fn,
                                                                compression=PZFFormat.DATA_COMP_HUFFCODE))

        # wait for the writer thread to process the queues (two flushes to be sure our puts were picked up)
        f.flush()
        f.flush()
        f.wait_close()
        assert not f._pollThread.is_alive()

        ds = HDFDataSource.DataSource(filename)
        try:
            assert ds.usePZFFormat
            assert list(ds.pzf_index['FrameNum']) == frame_nums
            assert ds.getNumSlices() == len(frame_nums)
            assert np.array_equal(ds.getSlice(4), frames[1])
        finally:
            ds.release()
    finally:
        shutil.rmtree(tempdir)


def test_h5r():
    data = tabular.ColumnSource(x=1e3*np.random.randn(1000), y=1e3*np.random.randn(1000), z=1e3*np.random.randn(1000))

    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_hdf.hdf')

    try:
        data.to_hdf(filename, tablename='Data')
    
        inp = tabular.HDFSource(filename, tablename='Data')
    
        assert (np.allclose(data['x'], inp['x']))
    finally:
        def wait_for_keepalive(func, path, exc_info):
            import time
            from PYME.IO.h5rFile import H5RFile
            time.sleep(H5RFile.KEEP_ALIVE_TIMEOUT + 1)
            func(path)
        # we can't delete the file until the thread releases it, so wait for the keep alive timeout
        shutil.rmtree(tempdir, onerror=wait_for_keepalive)
    