# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:21:06 2012

@author: David
"""

import eossdk as es
from ctypes import byref
class SDK(object):
    def __init__(self):
        es.EdsInitializeSDK()
        
    def __del__(self):
        es.EdsTerminateSDK()

#this should give us one global sdk object which gets terminated on exit        
_sdk = SDK()


class CanonEos(object):
    def __init__(self):
        cameraList = es.EdsCameraListRef()
        self.camera = es.EdsCameraRef()
        count = es.EdsUInt32(0)
        
        #get camera list        
        err = es.EdsGetCameraList(byref(cameraList))
        if not err == es.EDS_ERR_OK:
            raise RuntimeError('Failed to get camera list')
            
        #get number of cameras in list
        err = es.EdsGetChildCount(cameraList, byref(count))
        if count.value == 0:
            raise RuntimeError('No camera connected')
            
        #grab the first connected camera
        err = es.EdsGetChildAtIndex(cameraList, 0, byref(self.camera))
        if err:
            raise RuntimeError('EDSDK errno: %d' % err)
        
        #get device information
        deviceInfo = es.EdsDeviceInfo()
        err = es.EdsGetDeviceInfo(self.camera, byref(deviceInfo))
        
        if err:
            raise RuntimeError('EDSDK errno: %d' % err)
            
        #release camera list
        es.EdsRelease(cameraList)
        
        err = es.EdsOpenSession(self.camera)
        if err:
            raise RuntimeError('EDSDK errno: %d' % err)
        
    def __del__(self):
        #close session
        es.EdsCloseSession(self.camera)
        #release camera
        es.EdsRelease(self.camera)
        
        
        