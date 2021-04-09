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
        
        assert(np.allclose(im.data[:,:,:,0].squeeze(), data))
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
        
        assert (np.allclose(im.data[:, :, :, :].squeeze(), data))
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
        
def test_h5r():
    data = tabular.ColumnSource(x=1e3*np.random.randn(1000), y=1e3*np.random.randn(1000), z=1e3*np.random.randn(1000))

    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, 'test_hdf.hdf')

    try:
        data.to_hdf(filename, tablename='Data')
    
        inp = tabular.HDFSource(filename, tablename='Data')
    
        assert (np.allclose(data['x'], inp['x']))
    finally:
        shutil.rmtree(tempdir)
    