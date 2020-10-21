#!/usr/bin/python

##################
# init_TIRF.py
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


import time

@init_hardware('Camera')
def cam(scope):
    from PYME.Acquire.Hardware.uc480 import uCam480
    uCam480.init()
    cam = uCam480.uc480Camera(0, nbits=10)
    scope.register_camera(cam, 'Focus')
    scope.cam.SetGainBoost(False)  # shouldn't be needed, but make sure it is off
    scope.cam.SetGain(1)  # we really don't need any extra gain, this defaults to 10 on startup
    scope.cam.SetROI(289, 827, 1080, 1008)
    # Can't get frame rate higher than ~297 Hz for the current ROI, so default to just under that
    scope.cam.SetIntegTime(0.0035)  # [s]

#PIFoc
@init_hardware('PIFoc')
def pifoc(scope):
    from PYME.Acquire.Hardware.Piezos import offsetPiezoREST
    scope.piFoc = offsetPiezoREST.OffsetPiezoClient()
    scope.register_piezo(scope.piFoc, 'z')

# @init_gui('Raw Profile')
# def profile(MainFrame,scope):
#     from PYME.ui import fastGraph
#     import numpy as np
#
#     fg = fastGraph.FastGraphPanel(MainFrame, -1, np.arange(10), np.arange(10))
#     MainFrame.AddPage(page=fg, select=False, caption='Raw Profile')
#
#     def refr_profile(*args, **kwargs):
#
#         fg.SetData(np.arange(scope.frameWrangler.currentFrame.shape[1]), scope.frameWrangler.currentFrame.sum(0))
#
#     MainFrame.time1.WantNotification.append(refr_profile)

@init_gui('Focus Lock')
def focus_lock(MainFrame, scope):
    import numpy as np
    from PYME.ui import fastGraph
    from PYME.Acquire.Hardware.focus_locks.reflection_focus_lock import RLPIDFocusLockServer, FocusLogger
    from PYME.Acquire.ui.focus_lock_gui import FocusLockPanel, FocusLogPanel
    ku = -1.2  # ziegler-nichols 'ultimate' gain for my system
    tu = 7  # [frames], roughly the period, when we're running camera/frameWrangler polling at 3.5 ms
    # Stick with a PI tune for now
    kp = 0.45 * ku
    ki = 0.54 * ku / tu
    scope.focus_lock = RLPIDFocusLockServer(scope, scope.piFoc, p=kp, i=ki, d=0,
                                            sample_time=0.0035, 
                                            min_amp=10**5,
                                            max_sigma=20.)
    scope.focus_lock.register()
    panel = FocusLockPanel(MainFrame, scope.focus_lock)
    MainFrame.camPanels.append((panel, 'Focus Lock'))
    MainFrame.time1.WantNotification.append(panel.refresh)
    # we don't benefit at all from multiple frames piling up in a polling interval, so try and match the camera cycle
    scope.frameWrangler._polling_interval = 0.0035

    # # display dark-subtracted profile
    # fg = fastGraph.FastGraphPanel(MainFrame, -1, np.arange(10), np.arange(10))
    # MainFrame.AddPage(page=fg, select=False, caption='Profile')
    #
    # def refresh_profile(*args, **kwargs):
    #     profile = scope.frameWrangler.currentFrame.squeeze().sum(axis=0)
    #     if scope.focus_lock.subtraction_profile is not None:
    #         profile = profile - scope.focus_lock.subtraction_profile
    #     fg.SetData(np.arange(scope.frameWrangler.currentFrame.shape[1]), profile)
    #
    # MainFrame.time1.WantNotification.append(refresh_profile)

    # # display setpoint / error over time
    n = 500
    # setpoint = np.zeros(n)
    position = np.ones(n) * scope.focus_lock.peak_position
    time = np.arange(n)

    position_plot = fastGraph.FastGraphPanel(MainFrame, -1, time, position)
    MainFrame.AddPage(page=position_plot, select=False, caption='Position')
    #
    def refresh_position(*args, **kwargs):
        position[:-1] = position[1:]
        # if the position can't be found, replace nan with zero
        position[-1] = np.nan_to_num(scope.focus_lock.peak_position)
        # time[:-1] = time[1:]  # commenting out to leave time in units of frameWrangler.onFrameGroups for readability
        # time[-1] = scope.focus_lock._last_time
        position_plot.SetData(time, position)

    MainFrame.time1.WantNotification.append(refresh_position)

    # panel to log focus to file at set intervals
    focus_logger = FocusLogger(scope.focus_lock.GetPeakPosition)
    focus_log_panel = FocusLogPanel(MainFrame, focus_logger)
    MainFrame.camPanels.append((focus_log_panel, 'Focus Logger'))


#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

time.sleep(.5)
scope.initDone = True
