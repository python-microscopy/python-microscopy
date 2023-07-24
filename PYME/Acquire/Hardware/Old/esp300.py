#!/usr/bin/python

###############
# esp300.py
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
#LED Controll for TIRF

import example
import wx
from time import sleep
from math import *
import serial

class esp300:
    #To rotate the stage of the sw-tirf setup and
    #to switch a LED on/off
    #by ESP300 stepper motor controller ,type PR50PP
    

    def __init__(self, portname = 'COM4', maxtravel = (0.2, 0.2, 0.2)):
        self.max_travel = maxtravel
        self.ser_port = serial.Serial(portname, 19200, rtscts=0, timeout=4, writeTimeout=4)
        self.ser_port.write('1MO\r\n')
        self.ser_port.write('2MO\r\n')
        self.ser_port.write('3MO\r\n')
        self.ser_port.write('bo1h\r\n')
        self.ser_port.write('sb00h\r\n')
        #self.ser_port.write('ab\r\n')
        #pass
        
    def stop_all(self, event=0):
        #stop all axes (emergency)
        self.ser_port.write('AB\r')
        print('stop all axes')

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.ser_port.write('%dPA %1.2f\r\n' % (iChannel,fPos - self.max_travel[iChannel - 1]/2))
            else:
                self.ser_port.write('%dPA %1.2f\r\n' % (iChannel,self.max_travel[iChannel - 1]/2))
        else:
            self.ser_port.write('%dPA %1.2f\r\n' % (iChannel,0.00 - self.max_travel[iChannel - 1]/2))

    def GetPos(self, iChannel=1):
        self.ser_port.write('%dTP\r\n' % (iChannel,))
        res = self.ser_port.readline()
        return float(res) + self.max_travel[iChannel - 1]/2

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 1

    def GetMin(self,iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel[iChan - 1]
        
    def ledOn(self, event=0):
        #switch the LED on
        self.ser_port.write('sb01h\r\n')
        print('led on')
        
    def ledOff(self, event=1):
        # switch the LED off
        self.ser_port.write('sb00h\r\n')
        print('led off')        
    
    def addMenuItems(self,parentWindow, menu):
        """Add menu items and keyboard accelerators for LED control
        to the specified menu & parent window"""
        #Create IDs
        self.ID_stop_all = wx.NewIdRef()
        self.ID_LED_ON = wx.NewIdRef()
        self.ID_LED_OFF = wx.NewIdRef()
        
        #Add menu items
        menu.AppendSeparator()
        menu.Append(helpString='', id=self.ID_stop_all,
              item='EMERGENCY STOP', kind=wx.ITEM_NORMAL)
        menu.AppendSeparator()
        menu.Append(helpString='', id=self.ID_LED_ON,
              item='Led On', kind=wx.ITEM_NORMAL)
        menu.Append(helpString='', id=self.ID_LED_OFF,
              item='Led Off', kind=wx.ITEM_NORMAL)

        
        #Handle clicking on the menu items
        wx.EVT_MENU(parentWindow, self.ID_stop_all, self.stop_all)
        wx.EVT_MENU(parentWindow, self.ID_LED_ON, self.ledOn)
        wx.EVT_MENU(parentWindow, self.ID_LED_OFF, self.ledOff)
        
        