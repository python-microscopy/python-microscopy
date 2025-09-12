
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir(pure:false) / 'PYME/localization'
message('PYME/localization install dir: ', install_dir)

py_sources = files(
    'ofind3d.py',
    'splitting.py',
    'ofind.py',
    'ofind_nofilt.py',
    'MetaDataEdit.py',
    '__init__.py',
    'twist.py',
    'ofind_split.py',
    'remFitBuf.py',
    'ofind_pri.py',
    'setup.py',
    'ofind_xcorr_fw.py',
    'ofind_xcorr.py',
)

py.install_sources(py_sources, subdir:'PYME/localization')

subdir('cModels')
subdir('cInterp')
py_sources = files(
    'FitFactories/BeadConvolvedAstigGaussFit.py',
    'FitFactories/SplitterShiftEstFR.py',
    'FitFactories/ChecksumGPUFR.py',
    'FitFactories/SplitterROIExtractNR.py',
    'FitFactories/InterpFitR.py',
    'FitFactories/SplitterFitScavNR.py',
    'FitFactories/GaussFitConstrR.py',
    'FitFactories/fitCommon.py',
    'FitFactories/GaussMultifitR.py',
    'FitFactories/SplitterMultifitSR.py',
    'FitFactories/PRInterpFitQR.py',
    'FitFactories/Dumbell3DFitR.py',
    'FitFactories/SplitterFitCOIR.py',
    'FitFactories/ChecksumFR.py',
    'FitFactories/DumbellFitR.py',
    'FitFactories/MultiviewFitInterpNR.py',
    'FitFactories/ChecksumMultifitSR.py',
    'FitFactories/AstigGaussGPUFitFR.py',
    'FitFactories/AstigGaussFitFR.py',
    'FitFactories/SplitterObjFindR.py',
    'FitFactories/LatPSFFitR.py',
    'FitFactories/__init__.py',
    'FitFactories/SplitterFitFusionR.py',
    'FitFactories/Gauss3DFitR.py',
    'FitFactories/ROIExtractNR.py',
    'FitFactories/LatGaussFitFR.py',
    'FitFactories/FFBase.py',
    'FitFactories/GaussMultifitSR.py',
    'FitFactories/LatFitCOIR.py',
    'FitFactories/InterpFitWFR.py',
    'FitFactories/SplitterFitInterpNR.py',
    'FitFactories/SpecGaussFitFR.py',
    'FitFactories/PRInterpFitR.py',
    'FitFactories/SplitterFitFNR.py',
    'FitFactories/GaussMultifitLinesR.py',
    'FitFactories/LatObjFindFR.py',
    'FitFactories/SplitterFitInterpBNR.py',
    'FitFactories/ConfocCOIR.py',
)

py.install_sources(py_sources, subdir:'PYME/localization/FitFactories')

py_sources = files(
    'FitFactories/Interpolators/LinearInterpolator.py',
    'FitFactories/Interpolators/CubicSplineInterpolator.py',
    'FitFactories/Interpolators/__init__.py',
    'FitFactories/Interpolators/CSInterpolator.py',
    'FitFactories/Interpolators/LinearPInterpolatorP.py',
    'FitFactories/Interpolators/baseInterpolator.py',
    'FitFactories/Interpolators/CubeSInterpolatorP.py',
)

py.install_sources(py_sources, subdir:'PYME/localization/FitFactories/Interpolators')

py_sources = files(
    'FitFactories/zEstimators/biplaneEstimator.py',
    'FitFactories/zEstimators/genericDualEstimator.py',
    'FitFactories/zEstimators/__init__.py',
    'FitFactories/zEstimators/astigEstimator.py',
    'FitFactories/zEstimators/genericEstimator.py',
    'FitFactories/zEstimators/priEstimator.py',
)

py.install_sources(py_sources, subdir:'PYME/localization/FitFactories/zEstimators')

py_sources = files(
    'autopsf/autopsf.py',
    'autopsf/__init__.py',
    'autopsf/roifit.py',
)

py.install_sources(py_sources, subdir:'PYME/localization/autopsf')


# Test folder
# use a glob to find all test files - only way to install data files seems to be to pretend they are sources
gl = run_command(py, ['-c', '"import glob; print(glob.glob(\'PYME/localization/Test/*.*\'))"'], check: true)
test_files = gl.stdout().strip().split()
py.install_sources(test_files, subdir: 'PYME/localization/Test', pure:false, preserve_path: true)
#install_subdir('Test', install_dir: install_dir, strip_directory:s false)