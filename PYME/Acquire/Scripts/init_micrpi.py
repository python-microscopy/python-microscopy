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
"""
init file for light-sheet system running headless on a raspberry pi
"""

from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware

import time

@init_hardware('Cameras')
def cam(scope):
    from PYME.Acquire.Hardware.uc480 import uCam480
    uCam480.init(cameratype='ueye')
    cam = uCam480.uc480Camera(0)
    scope.register_camera(cam, 'ueye')



#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


#time.sleep(.5)
scope.initDone = True
