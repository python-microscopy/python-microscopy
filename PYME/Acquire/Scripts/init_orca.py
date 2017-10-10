#!/usr/bin/python

##################
# init.py
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

#!/usr/bin/python


from PYME.Acquire.Hardware import fakeShutters
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware
import scipy
import time
import os
import sys

fakeShutters = fakeShutters

def GetComputerName():
    if sys.platform == 'win32':
        return os.environ['COMPUTERNAME']
    else:
        return os.uname()[1]

InitBG('HamamatsuORCA', """
from PYME.Acquire.Hardware.HamamatsuDCAM import HamamatsuORCA
scope.cam = HamamatsuORCA.HamamatsuORCA(0)
scope.cameras['HamamatsuORCA'] = scope.cam

#time.sleep(5)
""")

InitGUI("""
scope.camControls['HamamatsuORCA'] = wx.Panel(MainFrame)
camPanels.append((scope.camControls['HamamatsuORCA'], 'ORCA Properties'))
""")

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    #hw = [fakeShutters.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo
#scope.shutters = fakeShutters

InitGUI("""scope.cam.Init()""")
#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#scope.SetCamera('A')

time.sleep(.5)
scope.initDone = True
