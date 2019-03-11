""" This just trys importing all the modules. Should cause test failures for py3k syntax errors (e.g. print statements)"""


def test_IO_imports():
    from PYME.IO import buffer_helpers, buffers
    from PYME.IO import clusterExport, clusterGlob
    from PYME.IO import clusterIO, clusterListing, clusterResults, clusterUpload
    from PYME.IO import dataExporter, dataWrap, dcimg, h5rFile
    from PYME.IO import image, load_psf, MetaDataHandler, PZFFormat
    from PYME.IO import ragged, tabular, unifiedIO, countdir
    
    #from PYME.IO import clusterDuplication #Known failure due to dependence on pyro