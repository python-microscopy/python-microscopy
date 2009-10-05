#!/usr/bin/python

##################
# NikonTE2000.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#import serial;
#import time
import win32com.client
nik = win32com.client.Dispatch("Nikon.TE2000.Microscope")

#from math import *
import eventLog

class zDrive:
    stepsize = 0.05 #stage moves in 50nm steps
    def __init__(self, maxtravel = 4500):
        self.hardMin = nik.ZDrive.Position.RangeLowerLimit
        self.hardMax = nik.ZDrive.Position.RangeHigherLimit
        self.max_travel = min(maxtravel, self.hardMax*self.stepsize)
	self.minorTick = self.stepsize*100 #set the slider tick length
        
    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        stepPos = round(fPos/self.stepsize)
        if (stepPos >= self.hardMin):
            if (fPos < self.max_travel):
                nik.ZDrive.Position = stepPos
            else:
                 nik.ZDrive.Position = round(self.max_travel/self.stepsize)
        else:
            nik.ZDrive.Position = self.hardMin
            
        eventLog.logEvent('Focus Change', 'New z-pos = %f' % stepPos)
    def GetPos(self, iChannel=1):
        return nik.ZDrive.Position.RawValue*self.stepsize
    def GetControlReady(self):
        return True
    def GetChannelObject(self):
        return 1
    def GetChannelPhase(self):
        return 1
    def GetMin(self,iChan=1):
        return self.hardMin*self.stepsize
        #return 3500
	#return round((self.GetPos() - 50)/50)*50
    
    def GetMax(self, iChan=1):
        return self.max_travel
	#return round((self.GetPos() + 50)/50)*50
