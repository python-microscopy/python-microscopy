
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/IO/FileUtils'

py_sources = files(
    'sf-thumbnailer.py',
    'IJLut.py',
    'h5rFormatDef.py',
    'fnlGen.py',
    'h5ExFrames.py',
    'obf_support.py',
    'thumbnailDatabase.py',
    'ply.py',
    '__init__.py',
    'readTiff.py',
    'freeSpace.py',
    'readSpeckle.py',
    'nameUtils.py',
    'fileID.py',
    'setup.py',
    'h5ToTiffSeries.py',
    'h5r-thumbnailer.py',
    'SPAD.py',
    'h5-thumbnailer.py',
    'TiffSeqToHdf5.py',
    'compHdf.py',
    'h5rToTxt.py',
    'RawToHdf5.py',
    'saveTiffStack.py',
    'stl.py',
)

py.install_sources(py_sources, subdir:'PYME/IO/FileUtils')

