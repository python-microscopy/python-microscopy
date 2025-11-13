
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
webui_static_files = files(
    'webui/static/css/pyme-bootstrap.css',
    'webui/static/css/pymeacquire.css',
    'webui/static/js/pymeacquire.js',
    'webui/static/js/pzf.js',
)
py.install_sources(webui_static_files, subdir:'PYME/Acquire', preserve_path: true)

webui_template_files = files(
    'webui/templates/PYMEAcquire.html',
    'webui/templates/login.html',
    'webui/templates/xtermpage.html',
)
py.install_sources(webui_template_files, subdir:'PYME/Acquire', preserve_path: true)

py_sources = files(
    'htsms/tweeter.py',
    'htsms/__init__.py',
    'htsms/rule_ui_v2.py',
    'htsms/rule_ui.py',
)
py.install_sources(py_sources, subdir:'PYME/Acquire/htsms')

scripts_files = files(
     'scripts/init.py',
     'scripts/init_N1_Ti_Exeter_ZIx.py',
     'scripts/init_N2_Ti_Zyla_lasers.py',
     'scripts/init_Neo.py',
     'scripts/init_NeoSim.py',
     'scripts/init_TIRF.py',
     'scripts/init_TIRF_Neo.py',
     'scripts/init_TIRF_NeoO.py',
     'scripts/init_TIRF_onecam.py',
     'scripts/init_Ti.py',
     'scripts/init_UOA_n.py',
     'scripts/init_UOA_n2.py',
     'scripts/init_Y1.py',
     'scripts/init_Zyla.py',
     'scripts/init_drift_tracking.py',
     'scripts/init_emccd_basic.py',
     'scripts/init_htsms.py',
     'scripts/init_htsms_focus_lock.py',
     'scripts/init_micrpi.py',
     'scripts/init_orca.py',
     'scripts/init_pco.py',
     'scripts/init_rev.py',
     'scripts/init_sim100.py',
     'scripts/init_sim103.py',
     'scripts/init_sim2.py',
     'scripts/init_sim50.py',
     'scripts/init_sim75.py',
     'scripts/init_sim_drift_tracking.py',
     'scripts/init_sim_htsms.py',
     'scripts/init_sim_htsms_n.py',
     'scripts/init_sim_main.py',
     'scripts/init_sim_min.py',
     'scripts/init_sim_rem.py',
     'scripts/init_smi1.py',
     'scripts/init_spectro.py',
     'scripts/init_spim.py',
     'scripts/init_thorcam.py',
     'scripts/init_twophoton.py',
     'scripts/init_uc480.py',
     'scripts/init_ueye.py',
)
py.install_sources(scripts_files, subdir:'PYME/Acquire/Scripts')
