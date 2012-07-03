#!/usr/bin/python

###############
# PYMEScreensaver.py
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
"""PYME Data Analysis Screensaver.
David Baddeley 2009
d.baddeley@auckland.ac.nz

Based on example code from the win32screensaver package by Chris Liechti
"""
import pyscr

import os
import subprocess
import sys
import time
import win32api

def cpuCount():
    '''
    Returns the number of CPUs in the system
    borrowed from the python 'processing' package
    '''
    if sys.platform == 'win32':
        try:
            num = int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            num = 0
    elif sys.platform == 'darwin':
        try:
            num = int(os.popen('sysctl -n hw.ncpu').read())
        except ValueError:
            num = 0
    else: #assuming unix
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            num = 0
        
    if num >= 1:
        return num
    else:
        raise NotImplementedError, 'cannot determine number of cpus'




class MySaver(pyscr.Screensaver):
    #set up timer for tick() calls
    TIMEBASE = 0.25

    def configure(self):
        #called to open the screensaver configuration
        #maybe implement here something with venster or win32all
        from ctypes import windll
        windll.user32.MessageBoxA(0,
            """Nothing to configure at present""",
            "PYME Screensaver Configuaration", 0, 0, 0)
    
    def initialize(self):
        #called once when the screensaver is started
        self.numProcessors = cpuCount()

        self.daughterProcesses = []

        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow=subprocess.SW_HIDE
        
        for i in range(self.numProcessors):
            #print 'launching daughter'
            self.daughterProcesses.append(subprocess.Popen('taskWorkerME.exe', startupinfo=info))
    
    def finalize(self):
        #called when the screensaver terminates

        #kill off all daughter processes
        for p in self.daughterProcesses:
            #p.kill() #only in 2.6
            handle = win32api.OpenProcess(1,False, p.pid)
            win32api.TerminateProcess(handle, -1)
            win32api.CloseHandle(handle)
    
    def tick(self):
        #called when the timer tick ocours, set up with startTimer from above
        self.dc.beginDrawing()

        w,h = self.dc.getSize()
        
        self.dc.setColor(0xffffffL)
        self.dc.setTextColor(0x00ff00L)
        self.dc.setBgColor(0x000000L)
        self.dc.setFont("courier new")
        #~ self.dc.setBgTransparent(True)
        #self.dc.drawLine((0+self.x, 0), (self.x*5, 50))
        self.dc.drawText((100,50), "Your computer has been assimilated into the...")
        self.dc.setTextColor(0xffffffL)
        self.dc.setFont("arial", 50, bold=True)
        self.dc.drawText((w/2 - 300,100), "PYthon Microscopy Environment")
        self.dc.setTextColor(0xff4f3fL)
        self.dc.setFont("arial", 250, bold=True)
        self.dc.drawText((w/2 - 250,150), "PYME")
        self.dc.setTextColor(0xffffffL)
        self.dc.setFont("arial", 50, bold=True)
        self.dc.drawText((w/2 - 300,400), "Distributed Data Analysis Collective")
        self.dc.setTextColor(0x00ff00L)
        self.dc.setFont("courier new")
        self.dc.drawText((100,470), "To exit move mouse or press a key")
        self.dc.drawText((100,490), "Queries: d.baddeley@auckland.ac.nz")

        self.dc.drawText((100,590), "%d CPUs detected" % self.numProcessors)
        
        self.dc.drawRect((0,0), (w-1,h-1))
        #self.x += 1
        #self.dc.fillEllipse((50, 50), (60, 60))
        self.dc.endDrawing()

#standard 'main' detection and startof screensaver
if __name__ == '__main__':
    pyscr.main()
