""" This just trys importing all the modules. Should cause test failures for py3k syntax errors (e.g. print statements)"""

import pytest
import sys

# mark some tests as expected to fail if we are testing on a headless system
try:
    import wx
    HAVE_WX = True
except ImportError:
    HAVE_WX = False

@pytest.mark.xfail(not HAVE_WX, reason="Depends on wx, which is not installed on this platform")
def test_IO_imports():
    from PYME.IO import buffer_helpers, buffers
    from PYME.IO import clusterExport, clusterGlob
    from PYME.IO import clusterIO, clusterListing, clusterResults, clusterUpload
    from PYME.IO import dataExporter, dataWrap, dcimg, h5rFile
    from PYME.IO import image, load_psf, MetaDataHandler, PZFFormat
    from PYME.IO import ragged, tabular, unifiedIO
    
    #from PYME.IO import clusterDuplication #Known failure due to dependence on pyro

@pytest.mark.skipif(sys.platform == 'win32', reason='countdir.c not compiled on non-posix systems')
def test_countdir_import():
    from PYME.IO import countdir
    
@pytest.mark.xfail(not HAVE_WX, reason="Depends on wx, which is not installed on this platform")
def test_DSView_imports():
    from PYME.DSView import arrayViewPanel, displayOptions, displaySettingsPanel, DisplayOptionsPanel
    from PYME.DSView import dsviewer, eventLogViewer, fitInfo, logparser, overlays, OverlaysPanel
    from PYME.DSView import scrolledImagePanel, splashScreen, voxSizeDialog
    
    from PYME.DSView import htmlServe # known fail due to cherrypy changes

@pytest.mark.xfail(not HAVE_WX, reason="Depends on wx, which is not installed on this platform")
def test_DSView_modules_imports():
    from PYME.DSView import modules

    modLocations = {}
    for m in modules.localmodules:
        modLocations[m] = ['PYME', 'DSView', 'modules']
    
    #import all modules
    for modName, ml in modLocations.items():
        __import__('.'.join(ml) + '.' + modName, fromlist=ml)

def test_c_imports():
    from PYME.Analysis.points.EdgeDB.edgeDB import addEdges, segment,calcEdgeLengths,getVertexEdgeLengths,getVertexNeighbours
    from PYME.Analysis.points.arcfit.arcmf import arcmf, arcmft, quad_surf_mf_fpos, quad_surf_mf
    from PYME.Analysis.points.astigmatism.astiglookup import astig_lookup
    from PYME.Analysis.points.DistHist.distHist import distanceHistogram, distanceHistogram3D, distanceProduct
    
@pytest.mark.xfail(not HAVE_WX, reason="Depends on wx, which is not installed on this platform")
def test_LMVis_imports():
    from PYME.LMVis import pipeline, renderers, gl_render3D_shaders
    from PYME.LMVis import visCore, visHelpers