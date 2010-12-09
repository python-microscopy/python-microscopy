#!/usr/bin/python

##################
# standard488.py
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

def setPos(pow):
    scope.l488.SetPower(pow)

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, setPos,10),
T(-1, scope.turnAllLasersOff),
T(20, setPos,10),
T(20, scope.l488.TurnOn),
T(1000, setPos,100),
T(2000, setPos,1000),
#T(20, scope.lFibre.TurnOn),
T(30, MainFrame.pan_spool.OnBAnalyse, None)
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 21)
]

#optional - pre-flight check
#a list of checks which should be performed prior to launching the protocol
#syntax: C(expression to evaluate (quoted, should have boolean return), message to display on failure),
preflight = [
C('scope.cam.GetEMGain() == 150', 'Was expecting an intial e.m. gain of 150'),
#C('scope.cam.GetROIX1() > 0', 'Looks like no ROI has been set'),
#C('scope.cam.GetIntegTime() <= 50', 'Camera integration time may be too long'),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, preflight, randomise = False)
