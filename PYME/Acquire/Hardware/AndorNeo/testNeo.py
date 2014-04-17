#!/usr/bin/python

###############
# testNeo.py
#
# Copyright David Baddeley, 2012
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
################


print 'Importing Camera ... '
import AndorNeo
import time
import numpy as np

print 'Initialising Camera ... '
cam = AndorNeo.AndorNeo(0)
cam.Init()

cam.SetIntegTime(100)
#cam.PixelReadoutRate.setIndex(2)

cam.SetROIIndex(7)
print 'Starting Exposure ...'
cam.StartExposure()

buf = np.empty((cam.GetPicHeight(),cam.GetPicWidth()), 'uint16')

print '\nStarting Extraction loop ...'
for i in range(20):
    print i,
    while cam.ExpReady():
        cam.ExtractColor(buf, 1)
        print 'e',
    
    time.sleep(.2)
        
time.sleep(20)

cam.Shutdown()
time.sleep(.5)
AndorNeo.camReg.unregCamera()

#import plotTimings
#plotTimings.PlotTimings()
        
        
        
        
