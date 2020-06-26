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


from PYME.Acquire.Hardware import fakeShutters
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware
import scipy
import time
import os
import sys

fakeShutters = fakeShutters


@init_hardware('ThorlabsCamera')
def thor_cam(scope):
    from PYME.Acquire.Hardware import thorlabs_cam
    cam = thorlabs_cam.ThorlabsCamera()
    
    scope.register_camera(cam, 'sCMOS')



# @init_gui('sCMOS Camera controls')
# def thor_cam_controls(MainFrame, scope):
#     import wx
#     # Generate an empty, dummy control panel
#     # TODO - adapt PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel or similar to allow options to be set.
#     # As it stands, we just use the default gain and readout settings.
#     scope.camControls['HamamatsuORCA'] = wx.Panel(MainFrame)
#     MainFrame.camPanels.append((scope.camControls['HamamatsuORCA'], 'ORCA Properties'))
    


#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


time.sleep(.5)
scope.initDone = True
