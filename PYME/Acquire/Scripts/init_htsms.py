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

from PYME.Acquire.ExecTools import joinBGInit, init_gui, init_hardware

from PYME import config
#enable high-throughput style directory hashing
config.config['acquire-spool_subdirectories'] = True

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
    scope.register_piezo(scope.stage, 'y', needCamRestart=False, channel=0, multiplier=1)

    scope.joystick = MarzHauserJoystick(scope.stage)
    scope.joystick.Enable(True)

    scope.CleanupFunctions.append(scope.stage.close)


@init_hardware('Z Piezo')
def pz(scope):
    from PYME.Acquire.Hardware.Piezos import piezo_e816_dll, offsetPiezoREST

    scope._piFoc = piezo_e816_dll.piezo_e816T(maxtravel=100)
    #scope.hardwareChecks.append(scope._piFoc.OnTarget)
    scope.CleanupFunctions.append(scope._piFoc.close)
    #scope.piFoc = scope._piFoc

    scope.piFoc = offsetPiezoREST.OffsetPiezoServer(scope._piFoc)
    scope.register_piezo(scope.piFoc, 'z', needCamRestart=False)

    from PYME.Acquire.Hardware.focus_locks.reflection_focus_lock import RLPIDFocusLockClient
    scope.focus_lock = RLPIDFocusLockClient()


@init_hardware('HamamatsuORCA')
def orca_cam(scope):
    from PYME.Acquire.Hardware.HamamatsuDCAM.HamamatsuORCA import MultiviewOrca

    size = 240
    multiview_info = {
        'Multiview.NumROIs': 4,
        'Multiview.ChannelColor': [0, 1, 1, 0],
        'Multiview.ROISize': (size, size),
        'Multiview.ROI0Origin': (104, 1024 - int(size / 2)),
        'Multiview.ROI1Origin': (844, 1024 - int(size / 2)),
        'Multiview.ROI2Origin': (1252, 1024 - int(size / 2)),
        'Multiview.ROI3Origin': (1724, 1024 - int(size / 2)),
    }
    cam = MultiviewOrca(0, multiview_info)
    cam.Init()

    # flip and rotate on primary camera should always be false - make the stage match the camera reference frame instead
    # as it's much easier
    # TODO - make flip, rotate etc actually work for tiling in case we have two cameras
    scope.register_camera(cam, 'HamamatsuORCA', rotate=False, flipx=False, flipy=False)

    def set_camera_views(views):
        if (views is None) or (len(views) == 0):
            cam.disable_multiview()
        else:
            cam.enable_multiview(views)

    scope.state.registerHandler('Camera.Views', lambda: cam.active_views, set_camera_views, True)


@init_gui('sCMOS Camera controls')
def orca_cam_controls(MainFrame, scope):
    import wx
    # Generate an empty, dummy control panel
    # TODO - adapt PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel or similar to allow options to be set.
    # As it stands, we just use the default gain and readout settings.
    scope.camControls['HamamatsuORCA'] = wx.Panel(MainFrame)
    MainFrame.camPanels.append((scope.camControls['HamamatsuORCA'], 'ORCA Properties'))

    # TODO - add a ROI / Views panel

    MainFrame.AddMenuItem('Camera', 'Set Multiview', lambda e: scope.state.setItem('Camera.Views', [0, 1, 2, 3]))
    MainFrame.AddMenuItem('Camera', 'Clear Multiview', lambda e: scope.state.setItem('Camera.Views', []))

    # from PYME.Acquire.ui import multiview_panel
    # mvp = multiview_panel.MultiviewPanel(MainFrame, scope)
    # MainFrame.camPanels.append((mvp, 'Multiview Panel'))


# @init_gui('Sample database')
# def samp_db(MainFrame, scope):
#     from PYME.Acquire import sampleInformation
#     sampPan = sampleInformation.slidePanel(MainFrame)
#     MainFrame.camPanels.append((sampPan, 'Current Slide'))

# @init_gui('Analysis settings')
# def anal_settings(MainFrame, scope):
#     from PYME.Acquire.ui import AnalysisSettingsUI
#     AnalysisSettingsUI.Plug(scope, MainFrame)


@init_hardware('Lasers & Shutters')
def lasers(scope):
    from PYME.Acquire.Hardware.Coherent import OBIS
    from PYME.Acquire.Hardware.MPBCommunications import MPBCW
    from PYME.Acquire.Hardware.AAOptoelectronics.MDS import AAOptoMDS
    from PYME.Acquire.Hardware.aotf import AOTFControlledLaser

    # key's are zero-indexed, MDS class does the same but add's one in commands when needed to match MDS API
    aotf_calibrations = {  # note this is a test dummy TODO - load from file
        0: {
            'wavelength': 405,  # nm
            'frequency': 154.3,  # MHz
            'aotf_setting': [
                18.3, 16, 14, 12, 10, 8.1, 0  # dBm
            ],
            'output': [ # note that 0 aotf_setting must correspond with 0 output setting
                1, 0.9, 0.7, 0.6, 0.4, 0.3, 0  # mW measured after objective
            ],
            'laser_setting': 100
        },
        1: {
            'wavelength': 488,  # nm
            'frequency': 115.614,  # MHz
            'aotf_setting': [
                0, 18.9, 18, 16, 14, 12, 10, 8, 6.1, 4, 2  # dBm
            ],
            'output': [  # note that 0 aotf_setting must correspond with 0 output setting
                0, 7.25, 6.9, 5.6, 4, 2.7, 1.8, 1, 0.6, 0.3, 0.2  # mW measured after objective
            ],
            'laser_setting': 45  # mW
        },
        2: {
            'wavelength': 560,  # nm
            'frequency': 94.820,  # MHz
            'aotf_setting': [
                20.1, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0  # dBm
            ],
            'output': [  # note that 0 aotf_setting must correspond with 0 output setting
                88.9, 77.5, 60, 41.7, 28.2, 18.4, 12.2, 7.9, 5.1, 3.5, 2.5  # mW measured after objective
            ],
            'laser_setting': 200
        },
        3: {
            'wavelength': 642,  # nm
            'frequency': 79.838,  # MHz
            'aotf_setting': [
                22.1, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0  # dBm
            ],
            'output': [ # note that 0 aotf_setting must correspond with 0 output setting
                82, 76.6, 61, 44.4, 30.6, 21.6, 14.9, 10.6, 7.8, 5.9, 4.8, 4.3  # mW measured after objective
            ],
            'laser_setting': 200
        },
    }

    from PYME.Acquire.Hardware.ioslave import FiberShaker
    # pins 3 and 5
    fiber_shaker = FiberShaker('COM9', channel=3, on_voltage=2.5)

    scope.aotf = AAOptoMDS(aotf_calibrations, 'COM14', 'AAOptoMDS', n_chans=4)
    scope.CleanupFunctions.append(scope.aotf.Close)

    l405 = OBIS.CoherentOBISLaser('COM10', name='OBIS405', turn_on=False)
    scope.CleanupFunctions.append(l405.Close)
    scope.l405 = AOTFControlledLaser(l405, scope.aotf, 0, chained_devices=[fiber_shaker])
    scope.l405.register(scope)

    l488 = OBIS.CoherentOBISLaser('COM13', name='OBIS488', turn_on=False)
    scope.CleanupFunctions.append(l488.Close)
    scope.l488 = AOTFControlledLaser(l488, scope.aotf, 1, chained_devices=[fiber_shaker])
    scope.l488.register(scope)

    l560 = MPBCW.MPBCWLaser('COM11', name='MPB560', turn_on=True,
                            init_power=200)  # minimum power for our MPB lasers is 200 mW
    scope.l560 = AOTFControlledLaser(l560, scope.aotf, 2, chained_devices=[fiber_shaker])
    scope.CleanupFunctions.append(scope.l560.Close)
    scope.l560.register(scope)

    l642 = MPBCW.MPBCWLaser('COM12', name='MPB642', turn_on=True,
                            init_power=200)  # minimum power for our MPB lasers is 200 mW
    scope.CleanupFunctions.append(l642.Close)
    scope.l642 = AOTFControlledLaser(l642, scope.aotf, 3, chained_devices=[fiber_shaker])
    scope.l642.register(scope)


@init_gui('Laser controls')
def laser_controls(MainFrame, scope):
    from PYME.Acquire.ui import lasersliders

    lsf = lasersliders.LaserSliders(MainFrame.toolPanel, scope.state)
    MainFrame.time1.WantNotification.append(lsf.update)
    MainFrame.camPanels.append((lsf, 'Laser Powers'))

# @init_hardware('Line scanner')
# def line_scanner(scope):
#     from PYME.experimental import scanner_control
#     scope.line_scanner = scanner_control.ScannerController()

# @init_gui('line scanner')
# def line_scanner_gui(MainFrame, scope):
#     from PYME.Acquire.ui import scanner_panel
#     from PYME.experimental import scanner_control
#
#     scope.line_scanner = scanner_control.ScannerController()
#     scp = scanner_panel.ScannerPanel(MainFrame.camPanel, scope.line_scanner)
#     MainFrame.camPanels.append((scp, 'Line Scanner'))

@init_gui('Focus Keys')
def focus_keys(MainFrame, scope):
    from PYME.Acquire.Hardware import focusKeys
    from PYME.Acquire.ui.focus_lock_gui import FocusLockPanel
    fk = focusKeys.FocusKeys(MainFrame, scope.piFoc)
    panel = FocusLockPanel(MainFrame, scope.focus_lock)
    MainFrame.camPanels.append((panel, 'Focus Lock'))
    MainFrame.time1.WantNotification.append(panel.refresh)

#splitter
# @init_gui('Splitter')
# def splitter(MainFrame, scope):
#     from PYME.Acquire.Hardware import splitter
#     splt1 = splitter.Splitter(MainFrame, scope, scope.cameras['EMCCD'], flipChan = 1, dichroic = 'Unspecified' ,
#                               transLocOnCamera = 'Top', flip=True, dir='up_down', constrain=False, cam_name='EMCCD')
#     splt2 = splitter.Splitter(MainFrame, scope, scope.cameras['sCMOS'], flipChan = 1, dichroic = 'FF700-Di01' ,
#                               transLocOnCamera = 'Right', flip=False, dir='left_right', constrain=False, cam_name='sCMOS')


#InitGUI("""
#from PYME.Acquire.Hardware import splitter
#splt = splitter.Splitter(MainFrame, None, scope, scope.cam)
#""")

@init_gui('Action manager')
def action_manager(MainFrame, scope):
    from PYME.Acquire.ui import actionUI

    ap = actionUI.ActionPanel(MainFrame, scope.actions, scope)
    MainFrame.AddPage(ap, caption='Queued Actions')

# @init_gui('Drift tracking')
# def drift_tracking(MainFrame, scope):
#     import subprocess
#     import sys
#     from PYME.Acquire import PYMEAcquire
#     scope.p_drift = subprocess.Popen('%s "%s" -i init_drift_tracking.py -t "Drift Tracking" -m "compact"' % (sys.executable, PYMEAcquire.__file__), shell=True)


#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


#time.sleep(.5)
scope.initDone = True


