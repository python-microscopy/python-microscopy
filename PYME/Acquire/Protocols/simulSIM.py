#!/usr/bin/python

##################
# simulPA.py
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
from PYME.Acquire.Hardware.Simulator import rend_im
import numpy as np

def SetPhase(ph):
    rend_im.SIM_phi = ph
    
def stop():
    MainFrame.pan_spool.OnBStopSpoolingButton(None)
    
def SetTheta(th):
    rend_im.SIM_theta = th

#set illumination period to 180 nm    
rend_im.SIM_k = np.pi/180
    
#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
#T(20, Ex, 'scope.cam.compT.laserPowers[0] = 1'),
T(-1, Ex, "scope.cam.compT.illumFcn = 'SIMIllumFcn'"), #Turn on SIM
T(-1, SetTheta, 0),
T(-1, SetPhase, 0),
T(1, SetPhase,  2*np.pi/3),
T(2, SetPhase,  4*np.pi/3),
T(3, SetTheta, np.pi/2),
T(3, SetPhase, 0),
T(4, SetPhase,  2*np.pi/3),
T(5, SetPhase,  4*np.pi/3),
T(6, SetTheta, np.pi/4),
T(6, SetPhase, 0),
T(7, SetPhase,  2*np.pi/3),
T(8, SetPhase,  4*np.pi/3),
T(9, SetTheta, -np.pi/4),
T(9, SetPhase, 0),
T(10, SetPhase,  2*np.pi/3),
T(11, SetPhase,  4*np.pi/3),
T(13, stop),
T(maxint, Ex, "scope.cam.compT.illumFcn = 'ConstIllum'")
#T(201, MainFrame.pan_spool.OnBAnalyse, None)
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (0, 20)),
('Protocol.DataStartsAt', 21)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, randomise = False)