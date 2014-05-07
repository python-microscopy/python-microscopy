#!/usr/bin/python

##################
# standard488.py
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

print(taskList)

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 21)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 30, metaData, randomise = False)
