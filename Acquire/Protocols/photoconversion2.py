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

def alternans(start, period, ncycles, chans):
    return [T(start + period*n, scope.dichroic.SetFilter, chans[n%2]) for n in range(2*ncycles)]

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, scope.turnAllLasersOff),
T(-1, SetCameraShutter, False),
T(-1, scope.dichroic.SetFilter, 'FITC'), #Pre-converted
T(20, SetCameraShutter, True),
T(30, MainFrame.pan_spool.OnBAnalyse, None),
]
taskList += alternans(40, 10, 5, ['Cy5', 'TxRed'])
taskList += [T(140, scope.dichroic.SetFilter, 'DAPI'),] #Photoconversion
taskList += alternans(250, 10, 5, ['Cy5', 'TxRed'])

print taskList

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 21)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 30, metaData, randomise = False)
