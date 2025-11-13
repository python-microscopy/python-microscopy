
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


# Coherent
py_sources = files(
    'Coherent/OBIS.py',
    'Coherent/Sapphire.py',
    'Coherent/__init__.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Coherent')

# GCS
py_sources = files(
    'GCS/GCS_DLL.py',
    'GCS/__init__.py',
    'GCS/gcs.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/GCS')

# HamamatsuDCAM
py_sources = files(
    'HamamatsuDCAM/HamamatsuDCAM.py',
    'HamamatsuDCAM/HamamatsuORCA.py',
    'HamamatsuDCAM/Hamamatsu_control_panel.py',
    'HamamatsuDCAM/__init__.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/HamamatsuDCAM')

# MPBCommunications
py_sources = files(
    'MPBCommunications/MPBCW.py',
    'MPBCommunications/__init__.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/MPBCommunications')

# Mercury
py_sources = files(
    'Mercury/Mercury.py',
    'Mercury/PI_Mercury_GCS_DLL.py',
    'Mercury/__init__.py',
    'Mercury/mercuryStepper.py',
    'Mercury/mercuryStepperGCS.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Mercury')

# Tango
py_sources = files(
    'Tango/__init__.py',
    'Tango/marzhauser_tango.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/Tango')

# arduino
# Only io_slave/ subdir present, skipping as no .py files at top level

# focus_locks
py_sources = files(
    'focus_locks/__init__.py',
    'focus_locks/reflection_focus_lock.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/focus_locks')

# pco
py_sources = files(
    'pco/__init__.py',
    'pco/pco_cam.py',
    'pco/pco_edge_42_lt.py',
    'pco/pco_sdk.py',
    'pco/pco_sdk_cam.py',
    'pco/pco_sdk_cam_control_panel.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/pco')

# uc480
py_sources = files(
    'uc480/AndorControlFrame.py',
    'uc480/__init__.py',
    'uc480/uCam480.py',
    'uc480/uc480.py',
    'uc480/uc480Deprecated.py',
    'uc480/uc480_h.py',
    'uc480/uc480_h_gen.py',
    'uc480/ucCamControlFrame.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/uc480')

# AAOptoelectronics
py_sources = files(
    'AAOptoelectronics/MDS.py',
    'AAOptoelectronics/__init__.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/AAOptoelectronics')

# ARCoptix
py_sources = files(
    'ARCoptix/__init__.py',
    'ARCoptix/lcdriver.py',
    'ARCoptix/lcserver32.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Hardware/ARCoptix')

