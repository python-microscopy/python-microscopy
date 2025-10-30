
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Acquire/Hardware'

py_sources = files(
    'driftTracking.py',
    'lasers.py',
    'NikonTiGUI.py',
    'CameraSkeleton.py',
    'thorlabs_cam.py',
    'NikonTi.py',
    'cobaltLaser.py',
    'thorlabsPiezo.py',
    'ccdAdjPanel.py',
    'mpd_picosecond_delayer.py',
    'matchboxLaser.py',
    'focusKeys.py',
    'TiLightCrafter.py',
    'ccdCalibrator.py',
    'multiview.py',
    'camera_noise.py',
    'phoxxLaser.py',
    'OrielCornerstone.py',
    'ueye.py',
    'splitter.py',
    '__init__.py',
    'fw102.py',
    'comports.py',
    'toptica_ibeam.py',
    'ExciterWheel.py',
    'driftTrackGUI.py',
    'phoxxLaserOLD.py',
    'priorarclampshutter.py',
    'NikonTE2000.py',
    'FocCorr.py',
    'thorlabs_mff_flipper.py',
    'FilterWheel.py',
    'microscope_adapter.py',
    'Camera.py',
    'setup.py',
    'arclampshutterpanel.py',
    'aotf.py',
    'cobaltLaser561.py',
    'cameraSoftwareBuffer.py',
    'PM100USB.py',
    'fakeShutters.py',
    'spacenav.py',
    'LaserControlFrame.py',
    'EMCCDTheory.py',
    'FocCorrR.py',
    'ioslave.py',
    'FrFilter.py',
    'DMDGui.py',
    'frZStage.py',
    'ids_peak_cam.py',
    'olympusix81.py',
    'priorLumen.py',
)

py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware')

py_sources = files(
    'AndorIXon/AndorCam.py',
    'AndorIXon/__init__.py',
    'AndorIXon/AndorControlFrame.py',
    'AndorIXon/AndorIXon.py',
)

py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/AndorIXon')

py_sources = files(
    'AndorNeo/ZylaControlPanel.py',
    'AndorNeo/SDK3Cam.py',
    'AndorNeo/SDK3.py',
    'AndorNeo/AndorNeoControlFrame.py',
    'AndorNeo/AndorZyla.py',
    'AndorNeo/__init__.py',
    'AndorNeo/plotTimings.py',
    'AndorNeo/AndorNeo.py',
    'AndorNeo/parseAtdebug.py',
    'AndorNeo/testNeo.py',
)

py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/AndorNeo')

py_sources = files(
    'DigiData/RemoteDigiData.py',
    'DigiData/DigiData.py',
    'DigiData/__init__.py',
    'DigiData/DigiDataClient.py',
    'DigiData/axDD132x.py',
)

py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/DigiData')

subdir('Simulator')

# py_sources = files(
#     'Old/mc2000.py',
#     'Old/esp300.py',
#     'Old/lpttest.py',
# )
# py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Old')

py_sources = files(
    'Piezos/piezo_e816.py',
    'Piezos/piezo_test.py',
    'Piezos/piezo_c867.py',
    'Piezos/piezo_e816_dll.py',
    'Piezos/piezo_pipython_gcs.py',
    'Piezos/__init__.py',
    'Piezos/offsetPiezoREST.py',
    'Piezos/offsetPiezo.py',
    'Piezos/piezo_e709.py',
    'Piezos/piezo_e255.py',
    'Piezos/base_piezo.py',
    'Piezos/piezo_e816b.py',
    'Piezos/piezo_e662.py',
    'Piezos/piezo_e816_corr.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Piezos')

# Install other hardware subdirs (Coherent, GCS, HamamatsuDCAM, MPBCommunications, Mercury, Tango, arduino, focus_locks, pco, uc480, AAOptoelectronics, ARCoptix)
# TODO: This hacks around  actually specifying the directories and files explicitly as we have done above
# Expedient for now, but in keeping with how you are "supposed" to use meson... 
install_subdir('Coherent', install_dir: install_dir / 'Coherent', strip_directory: true)
install_subdir('GCS', install_dir: install_dir / 'GCS', strip_directory: true)
install_subdir('HamamatsuDCAM', install_dir: install_dir / 'HamamatsuDCAM', strip_directory: true)
install_subdir('MPBCommunications', install_dir: install_dir / 'MPBCommunications', strip_directory: true)
install_subdir('Mercury', install_dir: install_dir / 'Mercury', strip_directory: true)
install_subdir('Tango', install_dir: install_dir / 'Tango', strip_directory: true)
install_subdir('arduino', install_dir: install_dir / 'arduino', strip_directory: true)
install_subdir('focus_locks', install_dir: install_dir / 'focus_locks', strip_directory: true)
install_subdir('pco', install_dir: install_dir / 'pco', strip_directory: true)
install_subdir('uc480', install_dir: install_dir / 'uc480', strip_directory: true)
install_subdir('AAOptoelectronics', install_dir: install_dir / 'AAOptoelectronics', strip_directory: true)
install_subdir('ARCoptix', install_dir: install_dir / 'ARCoptix', strip_directory: true)
