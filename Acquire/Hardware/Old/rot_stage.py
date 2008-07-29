#LED Controll for TIRF

import example
import wx
from time import sleep
from math import *
import serial

class rot_stage:
    #To rotate the stage of the sw-tirf setup 
    #by ESP300 stepper motor controller ,type SR50CC
    

    def __init__(self, portname = 'COM6'):
        self.ser_port = serial.Serial(portname, 19200, rtscts=0, timeout=4, writeTimeout=4)
        self.ser_port.write('3MO\r\n')
        self.ser_port.write('bo1h\r\n')
        self.ser_port.write('sb00h\r\n')
        #self.ser_port.write('ab\r\n')
        #pass
        
    def posa(self, event=0):
        #rotate to 0deg position
        self.ser_port.write('3PA0\r\n')
        print 'rotate to 0deg_pos'

    def posb(self, event=1):
        #rotate to +60deg position
        self.ser_port.write('3PA60\r\n')
        print 'rotate to +60deg_pos'

    def ledOn(self, event=0):
        #switch the LED on
        self.ser_port.write('sb01h\r\n')
        print 'led on'
        
    def ledOff(self, event=1):
        # switch the LED off
        self.ser_port.write('sb00h\r\n')
        print 'led off'        
    
    def addMenuItems(self,parentWindow, menu):
        '''Add menu items and keyboard accelerators for LED control
        to the specified menu & parent window'''
        #Create IDs
        self.ID_posa = wx.NewId()
        self.ID_posb = wx.NewId()
        self.ID_LED_ON = wx.NewId()
        self.ID_LED_OFF = wx.NewId()
        
        #Add menu items
        menu.AppendSeparator()
        menu.Append(helpString='', id=self.ID_posa,
              item='0 deg.', kind=wx.ITEM_NORMAL)

        menu.Append(helpString='', id=self.ID_posb,
              item='+60 deg.', kind=wx.ITEM_NORMAL)
        
        menu.AppendSeparator()
        menu.Append(helpString='', id=self.ID_LED_ON,
              item='Led On', kind=wx.ITEM_NORMAL)

        menu.Append(helpString='', id=self.ID_LED_OFF,
              item='Led Off', kind=wx.ITEM_NORMAL)

        
        #Handle clicking on the menu items
        wx.EVT_MENU(parentWindow, self.ID_posa, self.posa)
        wx.EVT_MENU(parentWindow, self.ID_posb, self.posb)
        wx.EVT_MENU(parentWindow, self.ID_LED_ON, self.ledOn)
        wx.EVT_MENU(parentWindow, self.ID_LED_OFF, self.ledOff)
        
        