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
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware


from PYME.Acquire.Hardware import fakeShutters
import time
import os
import sys

def GetComputerName():
    if sys.platform == 'win32':
        return os.environ['COMPUTERNAME']
    else:
        return os.uname()[1]

@init_hardware('Camera')
def cam(scope):
    from PYME.Acquire.Hardware.uc480 import uCam480
    uCam480.init()
    cam = uCam480.uc480Camera(0)
    scope.register_camera(cam, 'Focus')

#PIFoc
@init_hardware('PIFoc')
def pifoc(scope):
    from PYME.Acquire.Hardware.Piezos import offsetPiezoREST
    scope.piFoc = offsetPiezoREST.OffsetPiezoClient()
    scope.register_piezo(scope.piFoc, 'z')

@init_gui('Profile')
def profile(MainFrame,scope):
    from PYME.ui import fastGraph
    import numpy as np

    #xvs = np.arange(scope.frameWrangler.currentFrame.shape[1])

    fg = fastGraph.FastGraphPanel(MainFrame, -1, np.arange(10), np.zeros(10))
    MainFrame.AddPage(page=fg, select=False, caption='Profile')

    def refr_profile(*args, **kwargs):
        fg.SetData(np.arange(scope.frameWrangler.currentFrame.shape[1]), scope.frameWrangler.currentFrame.sum(0))

    scope.frameWrangler.onFrameGroup.connect(refr_profile)

# @init_gui('Drift tracking')
# def drift_tracking(MainFrame, scope):
#     from PYME.Acquire.Hardware import driftTracking, driftTrackGUI
#     scope.dt = driftTracking.correlator(scope, scope.piFoc)
#     dtp = driftTrackGUI.DriftTrackingControl(MainFrame, scope.dt)
#     MainFrame.camPanels.append((dtp, 'Focus Lock'))
#     MainFrame.time1.WantNotification.append(dtp.refresh)

# @init_gui('Focus Keys')
# def focus_keys(MainFrame, scope):
#     from PYME.Acquire.Hardware import focusKeys
#     fk = focusKeys.FocusKeys(MainFrame, scope.piezos[0])



#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#scope.SetCamera('A')

time.sleep(.5)
scope.initDone = True