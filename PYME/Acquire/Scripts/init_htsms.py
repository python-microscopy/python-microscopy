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

#!/usr/bin/python
from PYME.Acquire.ExecTools import joinBGInit, init_gui, init_hardware


@init_hardware('XY Stage')  # FIXME - may need module-level locks if we add 'x' and 'y' of the xy stage as different piezos
def mz_stage(scope):
    from PYME.Acquire.Hardware.Tango.marzhauser_tango import MarzhauserTangoXY, MarzHauserJoystick
    scope.stage = MarzhauserTangoXY()  # FIXME - change to threaded
    # scope.stage.SetSoftLimits(0, [1.06, 20.7])
    # scope.stage.SetSoftLimits(1, [.8, 17.6])

    scope.register_piezo(scope.stage, 'x', needCamRestart=False, channel=0, multiplier=1)
    scope.register_piezo(scope.stage, 'y', needCamRestart=False, channel=1, multiplier=1)

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

    # server so drift correction can connect to the piezo
    #pst = offsetPiezo.ServerThread(scope.piFoc)
    #pst.start()
    #scope.CleanupFunctions.append(pst.cleanup)


@init_hardware('HamamatsuORCA')
def orca_cam(scope):
    from PYME.Acquire.Hardware.HamamatsuDCAM.HamamatsuORCA import MultiviewOrca

    size = 240
    multiview_info = {
        'Multiview.NumROIs': 4,
        'Multiview.ROISize': (size, size),
        'Multiview.ROI0Origin': (104, 1024 - int(size/2)),
        'Multiview.ROI1Origin': (844, 1024 - int(size/2)),
        'Multiview.ROI2Origin': (1252, 1024 - int(size/2)),
        'Multiview.ROI3Origin': (1724, 1024 - int(size/2)),
    }
    cam = MultiviewOrca(0, multiview_info)
    cam.Init()

    scope.register_camera(cam, 'HamamatsuORCA', rotate=True, flipx=True, flipy=False)

    def set_camera_views(views):
        if (views is None) or (len(views) == 0):
            cam.disable_multiview()
        else:
            cam.enable_multiview(views)

    scope.state.registerHandler('Camera.Views', lambda : cam.active_views, set_camera_views, True)


@init_gui('sCMOS Camera controls')
def orca_cam_controls(MainFrame, scope):
    import wx
    # Generate an empty, dummy control panel
    # TODO - adapt PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel or similar to allow options to be set.
    # As it stands, we just use the default gain and readout settings.
    scope.camControls['HamamatsuORCA'] = wx.Panel(MainFrame)
    MainFrame.camPanels.append((scope.camControls['HamamatsuORCA'], 'ORCA Properties'))

    # TODO - add a ROI / Views panel

    MainFrame.AddMenuItem('Camera','Set Multiview', lambda e: scope.state.setItem('Camera.Views',[0,1,2,3]))
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

    scope.l405 = OBIS.CoherentOBISLaser('COM10', name='OBIS405')
    scope.CleanupFunctions.append(scope.l405.Close())

    scope.l488 = OBIS.CoherentOBISLaser('COM13', name='OBIS488')
    scope.CleanupFunctions.append(scope.l488.Close())

    scope.l560 = MPBCW.MPBCWLaser('COM11', name='MPB560', init_power=200)  # minimum power for our MPB lasers is 200 mW
    scope.CleanupFunctions.append(scope.l560.Close())

    scope.l642 = MPBCW.MPBCWLaser('COM12', name='MPB642', init_power=200)  # minimum power for our MPB lasers is 200 mW
    scope.CleanupFunctions.append(scope.l642.Close())

    scope.lasers = [scope.l405, scope.l488, scope.l560, scope.l642]

@init_gui('Laser controls')
def laser_controls(MainFrame, scope):
    from PYME.Acquire.ui import lasersliders

    lcf = lasersliders.LaserToggles(MainFrame.toolPanel, scope.state)
    MainFrame.time1.WantNotification.append(lcf.update)
    MainFrame.camPanels.append((lcf, 'Laser Control'))

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
    fk = focusKeys.FocusKeys(MainFrame, scope.piFoc)

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


