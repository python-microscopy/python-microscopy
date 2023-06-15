#!/usr/bin/python

###############
# mc2000.py
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
#mc2000 stepper controller
# 9600Baud 8N1 XonXoff

import example
import wx
import time
from math import *
import serial

class mc2000:
    def __init__(self, portname = 'COM5', maxtravel = (42, 34, 0)):
        self.max_travel = maxtravel
        self.ser_port = serial.Serial(portname, 9600, rtscts=0, xonxoff=1, timeout=1, writeTimeout=1)
        #pass

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.ser_port.flushOutput()
                time.sleep(0.1)
                if (iChannel == 1):
                    #self.ser_port.write('%2.5f 0 0 rmove \r\n' %(fPos - self.max_travel[iChannel - 1]/2))
                    print(('%2.5f 0 0 rmove \r\n' %(fPos - self.max_travel[iChannel - 1]/2)))
                else:
                    #self.ser_port.write('0 %2.5f 0 rmove \r\n' %(fPos - self.max_travel[iChannel - 1]/2))
                    print(('0 %2.5f 0 rmove \r\n' %(fPos - self.max_travel[iChannel - 1]/2)))
                    
            else:
                self.ser_port.flushOutput()
                time.sleep(0.05)
                # absolute move with X_Y_Z_move_\r\n
                # relative move with X_Y_Z_rmove_\r\n
                self.ser_port.write('0 0 0 rmove \r\n')                
        else:
            self.ser_port.flushOutput()
            time.sleep(0.05)
            self.ser_port.write('0 0 0 rmove \r\n')
            
    def GetPos(self, iChannel=1):
        self.ser_port.flushOutput()
        time.sleep(0.05)
        self.ser_port.write('p\r\n')
        self.ser_port.flushInput()
        time.sleep(0.05)
        #self.ser_port.write('%dTP\r\n' % (iChannel,))
        # here we have one line with xyz coordinate
        # but this ugly controller send not an EOL Sequence.
        # So we use a timeout of 0.1 !!!!
        resxyz = self.ser_port.readline()
        self.ser_port.flushInput()
        # now we separate for coordinate number i
        posxyz = resxyz.split(':')
        posipart = posxyz.__getitem__(iChannel)
        # cut the value from control sequences
        posi = posipart.split('\x1b')
        # and we have the value as string now
        ipos = posi.__getitem__(0)
        ipos = float(ipos) + self.max_travel[iChannel - 1]/2
        print(('Kanal: %d Wert: %2.5f' %(iChannel, ipos)))
        return ipos 
    
    def GetId(self):
        self.ser_port.flushOutput()
        time.sleep(0.05)
        self.ser_port.write('identify\r\n')
        self.ser_port.flushInput()
        time.sleep(0.05)
        res = self.ser_port.readline()
        self.ser_port.flushInput()
        time.sleep(0.05)
        return res

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

    def JoyOn(self, event=0):
        #switch the Joystick on
        self.ser_port.flushOutput()
        time.sleep(0.05)
        self.ser_port.write('1 j \r\n')
        print('joystick on')
        
    def JoyOff(self, event=1):
        # switch the Joystick off
        self.ser_port.flushOutput()
        time.sleep(0.05)
        self.ser_port.write('0 j \r\n')
        print('joystick off')        
    
    def addMenuItems(self,parentWindow, menu):
        """Add menu items and keyboard accelerators for joystick control
        to the specified menu & parent window"""
        #Create IDs
        self.ID_JOY_ON = wx.NewIdRef()
        self.ID_JOY_OFF = wx.NewIdRef()
        
        #Add menu items
        menu.AppendSeparator()
        menu.Append(helpString='', id=self.ID_JOY_ON,
              item='Joystick On', kind=wx.ITEM_NORMAL)
        menu.Append(helpString='', id=self.ID_JOY_OFF,
              item='Joystick Off', kind=wx.ITEM_NORMAL)

        #Handle clicking on the menu items
        wx.EVT_MENU(parentWindow, self.ID_JOY_ON, self.JoyOn)
        wx.EVT_MENU(parentWindow, self.ID_JOY_OFF, self.JoyOff)    
    

        
    
        