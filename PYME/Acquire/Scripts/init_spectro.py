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

#import scipy
#from Hardware.Simulator import fakeCam, fakePiezo, lasersliders, dSimControl
from Hardware.Spectrometer import specCam
from Hardware import fakeShutters
import time

from Hardware.OrielCornerstone import Cornerstone7400
import subprocess
import os

pz = InitBG('Monochromator', """
scope.monochromator = Cornerstone7400()
scope.monochromator.MoveTo(0,500)
scope.piezos.append((scope.monochromator, 1, 'Monochromator'))
scope.lasers = [scope.monochromator]
""")

cm = InitBG('Spectrometer', """
os.system('killall jythonOD') #kill off any previous spectrometer process
subprocess.Popen('jythonOD /home/david/PYME/PYME/Acquire/Hardware/Spectrometer/remoteSpectrometer.py', shell=True)
time.sleep(10) #wait for spectrometer process to start
scope.cam = specCam.SpecCamera()
#time.sleep(5)
""")

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [0] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo

scope.shutters = fakeShutters

#cm.join()


InitGUI("""
from PYME.Acquire.Hardware import LaserControlFrame
lcf = LaserControlFrame.LaserControlLight(MainFrame,scope.lasers)
time1.register_callback(lcf.refresh)
lcf.Show()
toolPanels.append((lcf, 'Laser Control'))
""")

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

scope.initDone = True


