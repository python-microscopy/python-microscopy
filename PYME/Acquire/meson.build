
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
#py = import('python').find_installation(pure: false)
#np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / 'PYME/Acquire'

py_sources = files(
    'frameWrangler.py',
    'stackSettings.py',
    'acquire_server.py',
    'microscope.py',
    'zScanner.py',
    'sampleInformation.py',
    'protocol.py',
    'PYMEAcquire.py',
    'acquire_client.py',
    'eventLog.py',
    'actions.py',
    '__init__.py',
    'SpoolController.py',
    'acquirebase.py',
    'protocol_acquisition.py',
    'event_loop.py',
    'xyztc.py',
    'setup.py',
    'ActionManager.py',
    'acquisition_base.py',
    'stage_leveling.py',
    'ExecTools.py',
    'positionTracker.py',
    'autofocus.py',
    'acquiremainframe.py',
    'acquirewx.py',
    'sampleInformationDjangoDirect.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire')

subdir('Hardware')

py_sources = files(
    'Protocols/tile.py',
    'Protocols/dual671_488.py',
    'Protocols/standard671.py',
    'Protocols/htsms-flow.py',
    'Protocols/sequence642_561.py',
    'Protocols/prebleach642.py',
    'Protocols/recover671.py',
    'Protocols/simulStep.py',
    'Protocols/prebleach561.py',
    'Protocols/shiftfield.py',
    'Protocols/simul488.py',
    'Protocols/darkCalibrate.py',
    'Protocols/strip_tile.py',
    'Protocols/simul642HTSMS.py',
    'Protocols/prebleach671ND1.py',
    'Protocols/simul642.py',
    'Protocols/htsms-two-color.py',
    'Protocols/prebleach561NeomEos2ND1.py',
    'Protocols/tile_triggered.py',
    'Protocols/prebleach488.py',
    'Protocols/prebleach561NeomEos2.py',
    'Protocols/sequence642_488.py',
    'Protocols/htsms-tile.py',
    'Protocols/tile671.py',
    'Protocols/htsms-widefield.py',
    'Protocols/__init__.py',
    'Protocols/spdscan.py',
    'Protocols/dual671_470.py',
    'Protocols/simulSIM.py',
    'Protocols/htsms-cal-registration.py',
    'Protocols/DMDMFM.py',
    'Protocols/htsms-cal-psf.py',
    'Protocols/standard488.py',
    'Protocols/photoconversion2.py',
    'Protocols/prebleach671ND2.py',
    'Protocols/photoconversion.py',
    'Protocols/prebleach490.py',
    'Protocols/prebleach671.py',
    'Protocols/htsms-staggered.py',
    'Protocols/prebleach671Neo.py',
    'Protocols/gainCal.py',
    'Protocols/standardArclamp.py',
    'Protocols/tile3D.py',
    'Protocols/simulPA.py',
    'Protocols/paint671.py',
    'Protocols/standard671_no532.py',
    'Protocols/standard470.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Protocols')

py_sources = files(
    'ui/focus_lock_gui.py',
    'ui/intensity_trace.py',
    'ui/AnalysisSettingsUI.py',
    'ui/splashScreen.py',
    'ui/selectCameraPanel.py',
    'ui/HDFSpoolFrame.py',
    'ui/__init__.py',
    'ui/tile_panel.py',
    'ui/intsliders.py',
    'ui/lasersliders.py',
    'ui/multiview_select.py',
    'ui/actionUI.py',
    'ui/mpd_picosecond_delay_panel.py',
    'ui/seqdialog.py',
    'ui/spool_panel.py',
    'ui/voxelSizeDialog.py',
    'ui/positionUI.py',
    'ui/tilesettingsui.py',
    'ui/preflight.py',
    'ui/scanner_panel.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/ui')

py_sources = files(
    'Utils/pointScanner.py',
    'Utils/__init__.py',
    'Utils/tiler.py',
    'Utils/MultiPointScanner.py',
    'Utils/failsafe.py',
    'Utils/strip_tiler.py',
    'Utils/vibrationAnalysis.py',
    'Utils/fastTiler.py',
    'Utils/sarcSpacing.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/Utils')

py_sources = files(
    'webui/__init__.py',
    'webui/ipy.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/webui')

py_sources = files(
    'htsms/tweeter.py',
    'htsms/__init__.py',
    'htsms/rule_ui_v2.py',
    'htsms/rule_ui.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/htsms')

install_subdir('Scripts', install_dir: install_dir )
