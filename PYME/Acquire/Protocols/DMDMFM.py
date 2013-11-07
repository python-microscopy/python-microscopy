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

def SetPattern(ph1, ph2):
    m = (((scope.LC.X + ph1) % 100) <= 3)*(((scope.LC.Y + ph2) % 100) <= 3)
    scope.LC.SetMask(m)    
    
def stop():
    MainFrame.pan_spool.OnBStopSpoolingButton(None)
    
#def SetTheta(th):
#    rend_im.SIM_theta = th

#set illumination period to 180 nm    
#rend_im.SIM_k = np.pi/180
    
#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
#T(20, Ex, 'scope.cam.compT.laserPowers[0] = 1'),
#T(-1, Ex, "scope.cam.compT.illumFcn = 'SIMIllumFcn'"), #Turn on SIM
#T(-1, SetTheta, 0),
T(-1, SetPattern, 0, 0),
T(1, SetPattern,  0, 2*np.pi/3),
T(2, SetPattern,  0, 4*np.pi/3),
T(3, SetPattern, np.pi/2, 0),
T(4, SetPattern, np.pi/2,  2*np.pi/3),
T(5, SetPattern, np.pi/2,  4*np.pi/3),
T(6, SetPattern, np.pi/4, 0),
T(7, SetPattern, np.pi/4,  2*np.pi/3),
T(8, SetPattern, np.pi/4,  4*np.pi/3),
T(9, SetPattern, -np.pi/4, 0),
T(10, SetPattern, -np.pi/4,  2*np.pi/3),
T(11, SetPattern, -np.pi/4,  4*np.pi/3),
#T(13, stop),
#T(maxint, Ex, "scope.cam.compT.illumFcn = 'ConstIllum'")
#T(201, MainFrame.pan_spool.OnBAnalyse, None)
]

#optional - metadata entries
metaData = [
#('Protocol.DarkFrameRange', (0, 20)),
#('Protocol.DataStartsAt', 21)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, randomise = False)