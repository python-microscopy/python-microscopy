#!/usr/bin/python

##################
# standard671.py
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
import numpy as np
import wx

nframes=200
INTEGRATIONTIMES = [5, 10, 25, 50, 100, 250]
TRANSITIONS = list(nframes*np.arange(len(INTEGRATIONTIMES)) + 20)

def stop():
    #ps.stop()
    wx.CallAfter(MainFrame.pan_spool.OnBStopSpoolingButton(None))

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [T(-1, scope.turnAllLasersOff),]
#taskList += [T(i,scope.cam.SetIntegTime, iTime) for i, iTime in zip(TRANSITIONS, INTEGRATIONTIMES)]
for i, iTime in zip(TRANSITIONS, INTEGRATIONTIMES):
	taskList += [T(i,scope.cam.SetIntegTime, iTime)]

# taskList += [T(10000, scope.filterWheel.SetFilterPos, "EMPTY"),]
taskList += [T(TRANSITIONS[-1]+nframes, stop),]

#optional - metadata entries
metaData = [
('Protocol.Transitions', TRANSITIONS),
('Protocol.IntegrationTimes', INTEGRATIONTIMES),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, randomise = False)
