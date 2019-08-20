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
    cam = uCam480.uc480Camera(0)
    scope.register_camera(cam, 'Focus')
    # try and hit about 40 Hz
    scope.cam.SetIntegTime(0.025)

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

    fg = fastGraph.FastGraphPanel(MainFrame, -1, np.arange(10), np.arange(10))
    MainFrame.AddPage(page=fg, select=False, caption='Profile')

    def refr_profile(*args, **kwargs):

        fg.SetData(np.arange(scope.frameWrangler.currentFrame.shape[1]), scope.frameWrangler.currentFrame.sum(0))

    MainFrame.time1.WantNotification.append(refr_profile)

@init_gui('Focus Lock')
def focus_lock(MainFrame, scope):
    from PYME.Acquire.Hardware.focus_locks.reflection_focus_lock import RLPIDFocusLockServer
    from PYME.Acquire.ui.focus_lock_gui import FocusLockPanel
    scope.focus_lock = RLPIDFocusLockServer(scope, scope.piFoc, p=0.01, i=0.0001, d=0.00005)
    scope.focus_lock.register()
    panel = FocusLockPanel(MainFrame, scope.focus_lock)
    MainFrame.camPanels.append((panel, 'Focus Lock'))
    MainFrame.time1.WantNotification.append(panel.refresh)



#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

time.sleep(.5)
scope.initDone = True
