#!/usr/bin/python

##################
# prebleach671.py
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

#import all the stuff to make this work
from PYME.Acquire.protocol import *
import numpy

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, scope.turnAllLasersOff),
T(-1, SetCameraShutter, False),
#T(-1, SetEMGain,150),
T(20, SetCameraShutter, True),
T(20, scope.filterWheel.SetFilterPos, "ND4"),
T(21, scope.l671.TurnOn),
T(58, scope.l671.TurnOff),
T(60, SetEMGain,0),
T(61, scope.l671.TurnOn),
T(61, scope.filterWheel.SetFilterPos, "ND1"),
T(101, SetEMGain,scope.cam.DefaultEMGain),
T(110, MainFrame.pan_spool.OnBAnalyse, None),
T(maxint, scope.turnAllLasersOff),
T(maxint, scope.filterWheel.SetFilterPos, "ND4"),
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 102),
('Protocol.PrebleachFrames', (21, 58)),
('Protocol.BleachFrames', (61,101)),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 101, 100, metaData, randomise = False)
