
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Analysis'

py_sources = files(
    'piecewiseMapping.py',
    'distributed_pyramid.py',
    'tile_pyramid.py',
    'widefieldify.py',
    'graphing_filters.py',
    'MetaData.py',
    'splitting.py',
    'fiducial_matching.py',
    'implicitFilter.py',
    'density_filter.py',
    'processLogger.py',
    'optic_flow.py',
    'svmSegment.py',
    'angleFilter.py',
    'gen_sCMOS_maps.py',
    '__init__.py',
    'thresholding.py',
    'annealThresh.py',
    '_fithelpers.py',
    'profile_extraction.py',
    'setup.py',
    'cramerRao.py',
    'piezo_movement_correction.py',
    'composite.py',
    'deTile.py',
    'piecewise.py',
    'keep_pairs.py',
    'binAvg.py',
    'autoencoder_classifier.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis')

subdir('points')
py_sources = files(
    'BleachProfile/rawIntensity.py',
    'BleachProfile/deMod.py',
    'BleachProfile/__init__.py',
    'BleachProfile/kinModels.py',
    'BleachProfile/intensProf.py',
    'BleachProfile/rollins.py',
    'BleachProfile/fitRecover.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/BleachProfile')

py_sources = files(
    'Colocalisation/__init__.py',
    'Colocalisation/edtColoc.py',
    'Colocalisation/colocScaleDep.py',
    'Colocalisation/correlationCoeffs.py',
    'Colocalisation/distColoc.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/Colocalisation')

py_sources = files(
    'Tracking/msd.py',
    'Tracking/__init__.py',
    'Tracking/trackUtils.py',
    'Tracking/tracking.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/Tracking')

py_sources = files(
    'PSFEst/extractImages.py',
    'PSFEst/fit_psf_zernikes.py',
    'PSFEst/psfQuality.py',
    'PSFEst/__init__.py',
    'PSFEst/dec.py',
)

py.install_sources(py_sources, subdir:'PYME/Analysis/PSFEst')

subdir('PSFGen')
