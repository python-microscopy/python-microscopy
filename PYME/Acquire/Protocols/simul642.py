#!/usr/bin/python

##################
# simul642.py
#
# This protocol is intended for use with the `init_sim_htsms.py` init script
# for a simulated high-throughput SMLM acquisition with cluster-based analysis.
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
T(-1, scope.state.update, {
        'Multiview.ActiveViews': [0, 1, 2, 3],
        'Multiview.ROISize': [256, 256],
        'Camera.IntegrationTime': 0.01,
    }),
T(20, scope.state.update, {'Lasers.l642.Power' : 1000, 'Lasers.l642.On' : True, }),
# T(30, MainFrame.pan_spool.OnBAnalyse, None),
T(maxint, scope.turnAllLasersOff)
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
#C('scope.cam.GetEMGain() == 150', 'Was expecting an intial e.m. gain of 150'),
#C('scope.cam.GetROIX1() > 0', 'Looks like no ROI has been set'),
#C('scope.cam.GetIntegTime() <= 50', 'Camera integration time may be too long'),
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, preflight, randomise = False)
