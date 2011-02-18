#!/usr/bin/python

##################
# prebleach671.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
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
T(-1, scope.joystick.Enable, False),
T(-1, SetCameraShutter, False),
T(20, SetCameraShutter, True),
#T(-1, SetEMGain,150),
T(20, scope.filterWheel.SetFilterPos, "ND2"),
T(21, scope.l488.TurnOn),
T(58, scope.l488.TurnOff),
T(60, SetEMGain,0),
T(61, scope.l488.TurnOn),
T(61, scope.filterWheel.SetFilterPos, "EMPTY"),
T(150, SetEMGain,scope.cam.DefaultEMGain),
T(160, MainFrame.pan_spool.OnBAnalyse, None),
T(maxint, scope.turnAllLasersOff),
T(maxint, scope.filterWheel.SetFilterPos, "ND2"),
T(maxint, scope.joystick.Enable, True),
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 151),
('Protocol.PrebleachFrames', (21, 58)),
('Protocol.BleachFrames', (61,150)),
]

#optional - pre-flight check
#a list of checks which should be performed prior to launching the protocol
#syntax: C(expression to evaluate (quoted, should have boolean return), message to display on failure),
preflight = [
C('scope.cam.GetEMGain() == scope.cam.DefaultEMGain', 'Was expecting an intial e.m. gain of 150'),
C('scope.cam.GetROIX1() > 1', 'Looks like no ROI has been set'),
C('scope.cam.GetIntegTime() < .06', 'Camera integration time may be too long'),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 101, 100, metaData, preflight, randomise = False)
