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
#T(-1, SetEMGain,150),
T(20, scope.filterWheel.SetFilterPos, "ND4.5"),
T(21, scope.l671.TurnOn),
T(58, scope.l671.TurnOff),
T(60, SetEMGain,0),
T(61, scope.l671.TurnOn),
T(61, scope.filterWheel.SetFilterPos, "EMPTY"),
T(150, SetEMGain,150),
T(160, MainFrame.pan_spool.OnBAnalyse, None),
T(maxint, scope.turnAllLasersOff),
T(maxint, scope.filterWheel.SetFilterPos, "ND4.5"),
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 151),
('Protocol.PrebleachFrames', (21, 58)),
('Protocol.BleachFrames', (61,150)),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 101, 100, metaData, randomise = False)
