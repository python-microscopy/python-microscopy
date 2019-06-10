# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:14:31 2013

@author: David Baddeley
"""

#space navigator 3D joystick

from pywinusb import hid
import numpy as np
import time

class SpaceNavigator(object):
    def __init__(self):
        #TODO - should be more specific here - there are likely to be cases 
        #where we have more than one HID device

        hid_dev = hid.find_all_hid_devices()

        sns = [h for h in hid_dev if (h.product_name == 'SpaceNavigator')]
        self.snav = sns[0]
        
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

    def close(self):
        self.snav.close()
            
    def __del__(self):
        self.close()
        
class SpaceNavPiezoCtrl(object):
    FULL_SCALE = 350.
    EVENT_RATE = 6.
    def __init__(self, spacenav, pz, pxy):
        self.spacenav = spacenav
        self.pz = pz#, self.px, self.py = piezos
        self.pxy = pxy
        
        self.xy_sensitivity = .01 #um/s
        self.z_sensitivity = -2 #um/s
        self.kappa = 1.5
        
        self.spacenav.WantPosNotification.append(self.updatePosition)
        self.spacenav.OnLeftButtonUp.append(self.leftButton)
        self.spacenav.OnRightButtonUp.append(self.rightButton)
        self.update_n= 0
        self.lastTime = 0
        
        
    def updatePosition(self, sn):
        sv = np.array([sn.x, sn.y, sn.z])/self.FULL_SCALE
        norm = np.linalg.norm(sv)
        if self.update_n % 10:
            
            #x_incr = float(sn.x*self.xy_sensitivity)/(self.FULL_SCALE*self.EVENT_RATE)
            #y_incr = float(sn.y*self.xy_sensitivity)/(self.FULL_SCALE*self.EVENT_RATE)
            z_incr = float(sv[2]*self.z_sensitivity)/(self.EVENT_RATE)
            
            
            #print x_incr, y_incr, z_incr
            #print sv, norm
    
            #try:
            if abs(sv[2]) >= norm/2:
                self.pxy.StopMove()
                try:
                    self.pz.MoveRel(0, z_incr)
                except:
                    pass
            #if abs(sn.x) >= norm/3:
            #    self.px[0].MoveRel(self.px[1], x_incr)
            #if abs(sn.y) >= norm/3:
            #    self.py[0].MoveRel(self.py[1], y_incr)
            #print sn.x/self.FULL_SCALE, sn.y/self.FULL_SCALE, sn.z/self.FULL_SCALE
            elif  (abs(sv[0]) >= norm/2 or abs(sv[1]) >= norm/2) and norm > .0001:
                t = time.time()
                dt = t - self.lastTime
                #print dt
                if True:#dt < .1: #constant motion
                    self.pxy.MoveInDir(self.xy_sensitivity*np.sign(sv[0])*abs(sv[0])**self.kappa, -self.xy_sensitivity*np.sign(sv[1])*abs(sv[1])**self.kappa)
                else: #nudge
                    xp, yp = self.pxy.GetPosXY()
                    
                    self.pxy.MoveToXY(xp + (abs(sv[0]) >= norm/2)*np.sign(sv[0])*.0003, yp - (abs(sv[1]) >= norm/2)*np.sign(sv[1])*.0003)
                    
                self.lastTime = t
            else:
                print( 's')
                self.pxy.StopMove()
                
            
            
            #except:
            #    pass
            
        self.update_n += 1

    
    def leftButton(self, sn):
        if self.xy_sensitivity < 0.005:
            self.xy_sensitivity *= 2
        print("spacenav speed: %d/6" %int(np.log2(self.xy_sensitivity/.001)+3))

    def rightButton(self, sn):
        if self.xy_sensitivity > 0.0004:
            self.xy_sensitivity /= 2
        print("spacenav speed: %d/6" %int(np.log2(self.xy_sensitivity/.001)+3))
