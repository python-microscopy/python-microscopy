#!/usr/bin/python

###############
# eos.py
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
        
        
        