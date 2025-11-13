
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/DSView'

py_sources = files(
    'OpenSequenceDialog.py',
    'splashScreen.py',
    'displaySettingsPanel.py',
    'logparser.py',
    'OverlaysPanel.py',
    '__init__.py',
    'setup.py',
    'overlays.py',
    'fitInfo.py',
    'arrayViewPanel.py',
    'htmlServe.py',
    'scrolledImagePanel.py',
    'displayOptions.py',
    'voxSizeDialog.py',
    'DisplayOptionsPanel.py',
    'eventLogViewer.py',
    'dsviewer.py',
)

py.install_sources(py_sources, subdir:'PYME/DSView')

subdir('LUT')
py_sources = files(
    'modules/camera_maps.py',
    'modules/_base.py',
    'modules/extractBlobs.py',
    'modules/annotation.py',
    'modules/playback.py',
    'modules/cropping.py',
    'modules/particleTracking.py',
    'modules/sheetView.py',
    'modules/convnet.py',
    'modules/LMAnalysis.py',
    'modules/prebleachExtraction.py',
    'modules/blobFinding.py',
    'modules/sparkprepare.py',
    'modules/eventView.py',
    'modules/filtering.py',
    'modules/deconvolution.py',
    'modules/shiftfield_estimation.py',
    'modules/fastGraphPanel.py',
    'modules/vis3D.py',
    'modules/shell.py',
    'modules/splitter.py',
    'modules/blobMeasure.py',
    'modules/flowView.py',
    'modules/psfGeneration.py',
    'modules/unmixing.py',
    'modules/__init__.py',
    'modules/graphViewPanel.py',
    'modules/synchronise.py',
    'modules/gl_image_view.py',
    'modules/object_tagging.py',
    'modules/fourierRingCorrelation.py',
    'modules/redimension.py',
    'modules/profileFitting.py',
    'modules/profilePlotting.py',
    'modules/coloc.py',
    'modules/metadataView.py',
    'modules/neuroglancer.py',
    'modules/LMDisplay.py',
    'modules/pupilTools.py',
    'modules/recipes.py',
    'modules/psfTools.py',
    'modules/composite.py',
    'modules/tiling.py',
    'modules/arrayView.py',
    'modules/webview.py',
    'modules/projections.py',
    'modules/psfExtraction.py',
    'modules/visgui.py',
    'modules/dark_and_flatfield.py',
    'modules/multithreshold.py',
    'modules/pupilView.py',
    'modules/segmentation.py',
)

py.install_sources(py_sources, subdir:'PYME/DSView/modules')

template_files = files(
    'modules/templates/astigCal.html',
    'modules/templates/measureView.html',
    'modules/templates/measureView2.html',
    'modules/templates/measureView3.html',
    'modules/templates/trackView.html',
    'modules/templates/tracksOverview.html',
)
py.install_sources(template_files, subdir: 'PYME/DSView/modules/templates', pure: false, preserve_path: false)

#modules/measure
measure_files = files(
    'modules/measure/__init__.py',
    'modules/measure/filterPan.py',
)
py.install_sources(measure_files, subdir:'PYME/DSView/modules/measure')
