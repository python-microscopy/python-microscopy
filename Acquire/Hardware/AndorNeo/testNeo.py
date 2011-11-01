# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:06:39 2011

@author: dbad004
"""

print 'Importing Camera ... '
import AndorNeo
import time
import numpy as np

print 'Initialising Camera ... '
cam = AndorNeo.AndorNeo(0)
cam.Init()

cam.SetIntegTime(100)
#cam.PixelReadoutRate.setIndex(2)

print 'Starting Exposure ...'
cam.StartExposure()

buf = np.empty((cam.GetPicWidth(), cam.GetPicHeight()), 'uint16')

print '\nStarting Extraction loop ...'
for i in range(200):
    print i,
    while cam.ExpReady():
        cam.ExtractColor(buf, 1)
        print 'e',
    
    time.sleep(.2)
        
time.sleep(20)

cam.Shutdown()
time.sleep(.5)
AndorNeo.camReg.unregCamera()

import plotTimings
plotTimings.PlotTimings()
        
        
        
        