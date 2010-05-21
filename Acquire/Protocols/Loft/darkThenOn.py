#!/usr/bin/python

##################
# darkThenOn.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#import all the stuff to make this work
from PYME.Acquire.protocol import *

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, scope.lasers['l488'].turnOff),
T(50, scope.lasers['l488'].turnOn)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList)