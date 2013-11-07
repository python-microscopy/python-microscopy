# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:14:31 2013

@author: David Baddeley
"""

#space navigator 3D joystick

from pywinusb import hid
import numpy as np

class SpaceNavigator(object):
    def __init__(self):
        #TODO - should be more specific here - there are likely to be cases 
        #where we have more than one HID device
        self.snav = hid.find_all_hid_devices()[-1]
        
        self.snav.open()
        
        self.x = 0
        self.y = 0
        self.z = 0
        self.rho = 0
        self.theta = 0
        self.phi = 0
        self.buttonState = 0
        
        self.WantPosNotification = []
        self.WantAngleNotification = []
        self.WantButtonNotification = []
        
        self.OnLeftButtonUp = []
        self.OnRightButtonUp = []
        
        self.snav.set_raw_data_handler(self._handleData)
        
    def _handleData(self, rawdata):
        if rawdata[0] == 1: #translation
            #print np.array(rawdata[1:], 'uint8').view('int16')
            self.x, self.y, self.z = np.array(rawdata[1:], 'uint8').view('int16')
            for cb in self.WantPosNotification:
                cb(self)
        elif rawdata[0] == 2: #rotation
            self.rho, self.theta, self.phi = np.array(rawdata[1:], 'uint8').view('int16')
            for cb in self.WantAngleNotification:
                cb(self)
        if rawdata[0] == 3: # buttons
            #self.x, self.y, self.z = np.array(rawdata[1:], 'uint8').view('int16')
            bs = rawdata[1]
            if self.buttonState == 1 and bs == 0: #clicked left button
                for cb in self.OnLeftButtonUp:
                    cb(self)
            if self.buttonState == 2 and bs == 0: #clicked left button
                for cb in self.OnRightButtonUp:
                    cb(self)
            self.buttonState = bs
            for cb in self.WantButtonNotification:
                cb(self)
        
class SpaceNavPiezoCtrl(object):
    FULL_SCALE = 350.
    EVENT_RATE = 6.
    def __init__(self, spacenav, pz, pxy):
        self.spacenav = spacenav
        self.pz = pz#, self.px, self.py = piezos
        self.pxy = pxy
        
        self.xy_sensitivity = .03 #um/s
        self.z_sensitivity = -2 #um/s
        self.kappa = 1.5
        
        self.spacenav.WantPosNotification.append(self.updatePosition)
        self.update_n= 0
        
        
    def updatePosition(self, sn):
        if self.update_n % 10:
            #x_incr = float(sn.x*self.xy_sensitivity)/(self.FULL_SCALE*self.EVENT_RATE)
            #y_incr = float(sn.y*self.xy_sensitivity)/(self.FULL_SCALE*self.EVENT_RATE)
            z_incr = float(sn.z*self.z_sensitivity)/(self.FULL_SCALE*self.EVENT_RATE)
            
            norm = (abs(sn.x) + abs(sn.y) + abs(sn.z))/self.FULL_SCALE
            #print x_incr, y_incr, z_incr
    
            #try:
            if abs(sn.z) >= norm/3:
                try:
                    self.pz.MoveRel(0, z_incr)
                except:
                    pass
            #if abs(sn.x) >= norm/3:
            #    self.px[0].MoveRel(self.px[1], x_incr)
            #if abs(sn.y) >= norm/3:
            #    self.py[0].MoveRel(self.py[1], y_incr)
            print sn.x/self.FULL_SCALE, sn.y/self.FULL_SCALE, sn.z/self.FULL_SCALE
            if  (abs(sn.x) >= norm/2 or abs(sn.y) >= norm/2) and norm > .0001:
                
                self.pxy.MoveInDir(self.xy_sensitivity*np.sign(sn.x)*abs(sn.x/self.FULL_SCALE)**self.kappa, -self.xy_sensitivity*np.sign(sn.y)*abs(sn.y/self.FULL_SCALE)**self.kappa)
                
            else:
                print 's'
                self.pxy.StopMove()
                
            #except:
            #    pass
            
        self.update_n += 1
            

        