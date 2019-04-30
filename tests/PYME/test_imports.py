""" This just trys importing all the modules. Should cause test failures for py3k syntax errors (e.g. print statements)"""


def test_IO_imports():
    from PYME.IO import buffer_helpers, buffers
    from PYME.IO import clusterExport, clusterGlob
    from PYME.IO import clusterIO, clusterListing, clusterResults, clusterUpload
    from PYME.IO import dataExporter, dataWrap, dcimg, h5rFile
    from PYME.IO import image, load_psf, MetaDataHandler, PZFFormat
    from PYME.IO import ragged, tabular, unifiedIO, countdir
    
    #from PYME.IO import clusterDuplication #Known failure due to dependence on pyro
    
def test_DSView_imports():
    from PYME.DSView import arrayViewPanel, displayOptions, displaySettingsPanel, DisplayOptionsPanel
    from PYME.DSView import dsviewer, eventLogViewer, fitInfo, logparser, overlays, OverlaysPanel
    from PYME.DSView import scrolledImagePanel, splashScreen, voxSizeDialog
    
    from PYME.DSView import htmlServe # known fail due to cherrypy changes

def test_DSView_modules_imports():
    from PYME.DSView import modules

    modLocations = {}
    for m in modules.localmodules:
        modLocations[m] = ['PYME', 'DSView', 'modules']
    
    #import all modules
    for modName, ml in modLocations.items():
        __import__('.'.join(ml) + '.' + modName, fromlist=ml)