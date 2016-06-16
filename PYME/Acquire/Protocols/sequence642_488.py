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

def stop():
    MainFrame.pan_spool.OnBStopSpoolingButton(None)

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, scope.turnAllLasersOff),
T(-1, SetCameraShutter, False),
T(-1, scope.joystick.Enable, False),
T(-1, scope.dichroic.SetFilter, '647LP'),
#T(-1, SetEMGain,150),
T(20, SetCameraShutter, True),
#T(20, scope.filterWheel.SetFilterPos, "ND4"),
T(20, scope.l642.SetPower, .01),
T(21, scope.l642.TurnOn),
T(58, scope.l642.TurnOff),
T(60, SetEMGain,0),
T(61, scope.l642.TurnOn),
T(61, scope.l642.SetPower, 1),
T(200, SetEMGain,scope.cam.DefaultEMGain),
T(210, MainFrame.pan_spool.OnBAnalyse, None),

T(10200, scope.l642.TurnOff),
T(10200, scope.dichroic.SetFilter, 'LF488'),
T(10200, scope.piFoc.MoveRel, 0, 0), #params = chan, offset - chan always zero.
T(10300, scope.l488.SetPower, 1),
T(10350, scope.l488.TurnOn),

T(20350, stop),
T(maxint, scope.turnAllLasersOff),
T(maxint, scope.l642.SetPower, .01),
T(maxint, scope.joystick.Enable, True),
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 201),
('Protocol.PrebleachFrames', (21, 58)),
('Protocol.BleachFrames', (61,200)),
('Protocol.NumberSequentialColors', 2),
('Protocol.ColorRange0', (200, 10200)),
('Protocol.ColorRange1', (10350, 20350)),
]

#optional - pre-flight check
#a list of checks which should be performed prior to launching the protocol
#syntax: C(expression to evaluate (quoted, should have boolean return), message to display on failure),
preflight = [
#C('scope.cam.GetEMGain() == scope.cam.DefaultEMGain', 'Was expecting an intial e.m. gain of %d' % scope.cam.DefaultEMGain),
C('scope.cam.GetROIX1() > 1', 'Looks like no ROI has been set'),
C('scope.cam.GetIntegTime() < .06', 'Camera integration time may be too long'),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 101, 100, metaData, preflight, randomise = False)
