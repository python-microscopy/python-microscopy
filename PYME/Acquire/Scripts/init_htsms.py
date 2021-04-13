#!/usr/bin/python

##################
# init.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

from PYME import config
from PYME.Acquire.ExecTools import joinBGInit, init_gui, init_hardware
#enable high-throughput style directory hashing
# config.config['acquire-spool_subdirectories'] = True

@init_hardware('XY Stage')  # FIXME - may need module-level locks if we add 'x' and 'y' of the xy stage as different piezos
def mz_stage(scope):
    from PYME.Acquire.Hardware.Tango.marzhauser_tango import MarzhauserTangoXY, MarzHauserJoystick
    scope.stage = MarzhauserTangoXY()  # FIXME - change to threaded
    # scope.stage.SetSoftLimits(0, [1.06, 20.7])
    # scope.stage.SetSoftLimits(1, [.8, 17.6])

    # the stage should match the camera reference frame - i.e. the 'x' channel should be the one which results in lateral
    # movement on the camera, and the y channel should result in vertical movement on the camera
    # multipliers should be set (+1 or -1) so that the direction also matches.
    scope.register_piezo(scope.stage, 'x', needCamRestart=False, channel=1, multiplier=1)
    scope.register_piezo(scope.stage, 'y', needCamRestart=False, channel=0, multiplier=-1)

    scope.joystick = MarzHauserJoystick(scope.stage)
    scope.joystick.Enable(True)

    scope.CleanupFunctions.append(scope.stage.close)


@init_hardware('Z Piezo')
def pz(scope):
    from PYME.Acquire.Hardware.Piezos import piezo_e816_dll, offsetPiezoREST as opr
    from PYME.Acquire.Hardware.focus_locks.reflection_focus_lock import RLPIDFocusLockClient
    from PYME.Acquire import stage_leveling, PYMEAcquire
    import sys
    import subprocess
    import requests

    # try and update the pifoc position roughly as often as the PID / camera, but a little faster if we can
    scope._piFoc = piezo_e816_dll.piezo_e816T(maxtravel=100, target_tol=0.035, update_rate=0.002)
    scope.CleanupFunctions.append(scope._piFoc.close)

    scope.piFoc = opr.generate_offset_piezo_server(opr.TargetOwningOffsetPiezo)(scope._piFoc)
    scope.register_piezo(scope.piFoc, 'z', needCamRestart=False)

    scope.focus_lock = RLPIDFocusLockClient()

    try:  # check if we've got a focus lock PYMEAcquire instance up already
        requests.get('http://127.0.0.1:9798/LockEnabled')
    except requests.exceptions.ConnectionError:
        fl_command = "%s" % PYMEAcquire.__file__
        fl_command += ' -i init_htsms_focus_lock.py -t "Focus Lock"'
        subprocess.Popen('%s %s' % (sys.executable, fl_command),
                        creationflags=subprocess.CREATE_NEW_CONSOLE)

    scope._stage_leveler = stage_leveling.StageLeveler(scope, scope.piFoc,
                                                       focus_lock=scope.focus_lock,
                                                       pause_on_relocate=1.0)


@init_hardware('HamamatsuORCA')
def orca_cam(scope):
    from PYME.Acquire.Hardware.HamamatsuDCAM.HamamatsuORCA import MultiviewOrca

    # centers, (x [0, 2047], y [0, 400]), 2019/08/14
    # [(291, 199),
    # (857, 199),
    # (1257, 199),
    # (1795, 198)
    # ]

    size = 256
    half_size = int(size / 2)
    multiview_info = {
        'Multiview.NumROIs': 4,
        'Multiview.ChannelColor': [0, 1, 1, 0],
        'Multiview.DefaultROISize': (size, size),
        'Multiview.ROISizeOptions': [128, 240, 256, 304, 352, 384],
        'Multiview.ROI0Origin': (312 - half_size, 1024 - half_size),
        'Multiview.ROI1Origin': (876 - half_size, 1024 - half_size),
        'Multiview.ROI2Origin': (1268 - half_size, 1024 - half_size),
        'Multiview.ROI3Origin': (1744 - half_size, 1024 - half_size),
    }
    cam = MultiviewOrca(0, multiview_info)
    cam.Init()

    # flip and rotate on primary camera should always be false - make the stage match the camera reference frame instead
    # as it's much easier
    # TODO - make flip, rotate etc actually work for tiling in case we have two cameras
    scope.register_camera(cam, 'HamamatsuORCA', rotate=False, flipx=False, flipy=False)
    cam.register_state_handlers(scope.state)


@init_gui('sCMOS Camera controls')
def orca_cam_controls(MainFrame, scope):
    import wx
    # Generate an empty, dummy control panel
    # TODO - adapt PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel or similar to allow options to be set.
    # As it stands, we just use the default gain and readout settings.
    scope.camControls['HamamatsuORCA'] = wx.Panel(MainFrame)
    MainFrame.camPanels.append((scope.camControls['HamamatsuORCA'], 'ORCA Properties'))

    MainFrame.AddMenuItem('Camera', 'Set Multiview', 
                          lambda e: scope.state.setItem('Multiview.ActiveViews', [0, 1, 2, 3]))
    MainFrame.AddMenuItem('Camera', 'Clear Multiview', 
                          lambda e: scope.state.setItem('Multiview.ActiveViews', []))

@init_gui('Sample Metadata')
def sample_metadata(main_frame, scope):
    from PYME.Acquire.sampleInformation import SimpleSampleInfoPanel
    sampanel = SimpleSampleInfoPanel(main_frame)
    main_frame.camPanels.append((sampanel, 'Sample Metadata'))

@init_hardware('Lasers & Shutters')
def lasers(scope):
    from PYME.Acquire.Hardware.Coherent import OBIS
    from PYME.Acquire.Hardware.MPBCommunications import MPBCW
    from PYME.Acquire.Hardware.AAOptoelectronics.MDS import AAOptoMDS
    from PYME.Acquire.Hardware.aotf import AOTFControlledLaser
    from PYME.config import config
    import json

    calib_file = config['aotf-calibration-file']
    with open(calib_file, 'r') as f:
        aotf_calibration = json.load(f)

    scope.aotf = AAOptoMDS(aotf_calibration, 'COM14', 'AAOptoMDS', n_chans=4)
    scope.CleanupFunctions.append(scope.aotf.Close)

    # fiber_shaker = ServoFiberShaker('COM9', channel=9, on_value=50)  # pin 9

    l405 = OBIS.CoherentOBISLaser('COM10', name='OBIS405', turn_on=False)
    scope.CleanupFunctions.append(l405.Close)
    scope.l405 = AOTFControlledLaser(l405, scope.aotf, 0)  # , chained_devices=[fiber_shaker])
    scope.l405.register(scope)

    l488 = OBIS.CoherentOBISLaser('COM13', name='OBIS488', turn_on=False)
    scope.CleanupFunctions.append(l488.Close)
    scope.l488 = AOTFControlledLaser(l488, scope.aotf, 1)  # , chained_devices=[fiber_shaker])
    scope.l488.register(scope)

    l560 = MPBCW.MPBCWLaser('COM11', name='MPB560', turn_on=True,
                            init_power=200)  # minimum power for our MPB lasers is 200 mW
    scope.l560 = AOTFControlledLaser(l560, scope.aotf, 2)  # ,  chained_devices=[fiber_shaker])
    scope.CleanupFunctions.append(scope.l560.Close)
    scope.l560.register(scope)

    l642 = MPBCW.MPBCWLaser('COM12', name='MPB642', turn_on=True,
                            init_power=200)  # minimum power for our MPB lasers is 200 mW
    scope.CleanupFunctions.append(l642.Close)
    scope.l642 = AOTFControlledLaser(l642, scope.aotf, 3)  # ,  chained_devices=[fiber_shaker])
    scope.l642.register(scope)


@init_gui('Laser controls')
def laser_controls(MainFrame, scope):
    from PYME.Acquire.ui import lasersliders

    lsf = lasersliders.LaserSliders(MainFrame.toolPanel, scope.state)
    MainFrame.time1.WantNotification.append(lsf.update)
    MainFrame.camPanels.append((lsf, 'Laser Powers'))

@init_gui('Failsafe')
def failsafe(MainFrame, scope):
    from PYME import config
    from PYME.Acquire.Utils.failsafe import FailsafeServer
    import yaml

    email_info = config.get('email-info-path')
    with open(email_info, 'r') as f:
        email_info = yaml.safe_load(f)

    address = config.get('failsafeserver-address', '127.0.0.1')
    port = config.get('failsafeserver-port', 9119)
    scope.failsafe = FailsafeServer(scope, email_info, port, address)

@init_gui('Multiview Selection')
def multiview_selection(MainFrame, scope):
    from PYME.Acquire.ui import multiview_select

    ms = multiview_select.MultiviewSelect(MainFrame.toolPanel, scope)
    MainFrame.time1.WantNotification.append(ms.update)
    MainFrame.camPanels.append((ms, 'Multiview Selection'))

@init_gui('Focus Keys')
def focus_keys(MainFrame, scope):
    from PYME.Acquire.Hardware import focusKeys
    from PYME.Acquire.ui.focus_lock_gui import FocusLockPanel
    fk = focusKeys.FocusKeys(MainFrame, scope.piFoc)
    panel = FocusLockPanel(MainFrame, scope.focus_lock, offset_piezo=scope.piFoc)
    MainFrame.camPanels.append((panel, 'Focus Lock'))
    MainFrame.time1.WantNotification.append(panel.refresh)

@init_gui('Action manager')
def action_manager(MainFrame, scope):
    from PYME import config
    from PYME.Acquire.ui import actionUI
    from PYME.Acquire.ActionManager import ActionManagerServer

    ap = actionUI.ActionPanel(MainFrame, scope.actions, scope)
    MainFrame.AddPage(ap, caption='Queued Actions')

    ActionManagerServer(scope.actions, 9393, 
                        config.get('actionmanagerserver-address', '127.0.0.1'))

@init_gui('Chained Analysis')
def chained_analysis(main_frame, scope):
    from PYME.Acquire.htsms.rule_ui import SMLMChainedAnalysisPanel, get_rule_tile, RuleChain
    from PYME.cluster.rules import RecipeRuleFactory, SpoolLocalLocalizationRuleFactory
    from PYME.IO.MetaDataHandler import DictMDHandler
    import yaml
    import os

    # add some default pairings
    defaults = {}
    rec_dir = 'C:\\Users\\Bergamot\\PYMEData\\recipes'

    tilerec = os.path.join(rec_dir, '20210125_tile_detect_filter_queue_subset.yaml')
    with open(tilerec) as f:
        tilerec = f.read()
    defaults['htsms-tile'] = RuleChain([get_rule_tile(RecipeRuleFactory)(recipe=tilerec)])

    mdh = DictMDHandler({
            "Analysis.BGRange": [-32, 0],
            "Analysis.DebounceRadius": 4,
            "Analysis.DetectionFilterSize": 4,
            "Analysis.DetectionThreshold": 1.0,
            "Analysis.FiducialThreshold": 1.8,
            "Analysis.FitModule": "AstigGaussGPUFitFR",
            "Analysis.GPUPCTBackground": True,
            "Analysis.PCTBackground": 0.25,
            "Analysis.ROISize": 7.5,
            "Analysis.StartAt": 32,
            "Analysis.TrackFiducials": False,
            "Analysis.subtractBackground": True,
    })

    defaults['htsms-flow'] = RuleChain([get_rule_tile(SpoolLocalLocalizationRuleFactory)(analysisMetadata=mdh)])
    defaults['htsms-staggered'] = RuleChain([get_rule_tile(SpoolLocalLocalizationRuleFactory)(analysisMetadata=mdh)])

    SMLMChainedAnalysisPanel.plug(main_frame, scope, defaults)

@init_gui('Tiling')
def tiling(MainFrame, scope):
    from PYME.Acquire.ui import tile_panel

    ap = tile_panel.CircularTilePanel(MainFrame, scope)
    MainFrame.aqPanels.append((ap, 'Circular Tile Acquisition'))

    # ap = tile_panel.MultiwellTilePanel(MainFrame, scope)
    ap = tile_panel.MultiwellProtocolQueuePanel(MainFrame, scope)
    MainFrame.aqPanels.append((ap, 'Multiwell Tile Acquisition'))

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


#time.sleep(.5)
scope.initDone = True
