#!/usr/bin/python

##################
# init_TIRF.py
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

from PYME.Acquire.Hardware.uc480 import uCam480

from PYME.Acquire.Hardware import fakeShutters
import time
import os
import sys

def GetComputerName():
    if sys.platform == 'win32':
        return os.environ['COMPUTERNAME']
    else:
        return os.uname()[1]

#scope.cameras = {}
#scope.camControls = {}
from PYME.IO import MetaDataHandler

InitBG('EMCCD Cameras', """
scope.cameras['A - Left'] = uCam480.uc480Camera(2)
#scope.cameras['B - Right'] = AndorIXon.iXonCamera(0)
#scope.cameras['B - Right'].SetShutter(False)
#scope.cameras['B - Right'].SetActive(False)
scope.cam = scope.cameras['A - Left']
""")


#PIFoc
InitBG('PIFoc', """
from PYME.Acquire.Hardware.Piezos import offsetPiezo
scope.piFoc = offsetPiezo.getClient()
scope.register_piezo(scope.piFoc, 'z')
""")

InitGUI("""
from PYME.Acquire.Hardware import driftTracking, driftTrackGUI
scope.dt = driftTracking.correlator(scope, scope.piFoc)
dtp = driftTrackGUI.DriftTrackingControl(MainFrame, scope.dt)
camPanels.append((dtp, 'Focus Lock'))
time1.register_callback(dtp.refresh)
""", 'Drift Tracking')




#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#scope.SetCamera('A')

time.sleep(.5)
scope.initDone = True
