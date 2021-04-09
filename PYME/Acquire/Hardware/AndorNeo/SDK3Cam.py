#!/usr/bin/python

###############
# SDK3Cam.py
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


from .import SDK3

class ATProperty(object):
    def connect(self, handle, propertyName):
        self.handle = handle
        self.propertyName = propertyName
    def isImplemented(self):
        return SDK3.IsImplemented(self.handle, self.propertyName).value
    def isReadable(self):
        return SDK3.IsReadable(self.handle, self.propertyName).value
    def isWritable(self):
        return SDK3.IsWritable(self.handle, self.propertyName).value
    def isReadOnly(self):
        return SDK3.IsReadOnly(self.handle, self.propertyName).value


class ATInt(ATProperty):
    def getValue(self):
        return SDK3.GetInt(self.handle, self.propertyName).value
    
    def setValue(self, val):
        SDK3.SetInt(self.handle, self.propertyName, val)
        
    def max(self):
        return SDK3.GetIntMax(self.handle, self.propertyName).value
        
    def min(self):
        return SDK3.GetIntMin(self.handle, self.propertyName).value
        

class ATBool(ATProperty):
    def getValue(self):
        return SDK3.GetBool(self.handle, self.propertyName).value > 0
    
    def setValue(self, val):
        SDK3.SetBool(self.handle, self.propertyName, val)
    

        
class ATFloat(ATProperty):
    def getValue(self):
        return SDK3.GetFloat(self.handle, self.propertyName).value
    
    def setValue(self, val):
        SDK3.SetFloat(self.handle, self.propertyName, val)
        
    def max(self):
        return SDK3.GetFloatMax(self.handle, self.propertyName).value
        
    def min(self):
        return SDK3.GetFloatMin(self.handle, self.propertyName).value
        
class ATString(ATProperty):
    def getValue(self):
        return SDK3.GetString(self.handle, self.propertyName, 255).value
    
    def setValue(self, val):
        SDK3.SetString(self.handle, self.propertyName, val)
        
    def maxLength(self):
        return SDK3.GetStringMaxLength(self.handle, self.propertyName).value
        
class ATEnum(ATProperty):
    def getIndex(self):
        return SDK3.GetEnumIndex(self.handle, self.propertyName).value
    
    def setIndex(self, val):
        SDK3.SetEnumIndex(self.handle, self.propertyName, val)
        
    def getString(self):
        return self.__getitem__(self.getIndex())
    
    def setString(self, val):
        SDK3.SetEnumString(self.handle, self.propertyName, val)
        
    def __len__(self):
        return SDK3.GetEnumCount(self.handle, self.propertyName).value
    
    def __getitem__(self, key):         
        return SDK3.GetEnumStringByIndex(self.handle, self.propertyName, key, 255).value
        
    def getAvailableValues(self):
        n = SDK3.GetEnumCount(self.handle, self.propertyName).value
        
        return [SDK3.GetEnumStringByIndex(self.handle, self.propertyName, i, 255).value for i in range(n) if SDK3.IsEnumIndexAvailable(self.handle, self.propertyName, i).value]
        
class ATCommand(ATProperty):
    def __call__(self):
        return SDK3.Command(self.handle, self.propertyName)
        
class camReg(object):
    #keep track of the number of cameras initialised so we can initialise and finalise the library
    numCameras = 0
    
    @classmethod
    def regCamera(cls):
        if cls.numCameras == 0:
            SDK3.InitialiseLibrary()
        
        cls.numCameras += 1
        
    @classmethod
    def unregCamera(cls):
        cls.numCameras -= 1
        if cls.numCameras == 0:
            SDK3.FinaliseLibrary()
            
#make sure the library is intitalised
camReg.regCamera()

def GetNumCameras():
    return SDK3.GetInt(SDK3.AT_HANDLE_SYSTEM, 'DeviceCount').value
    
def GetSoftwareVersion():
    return SDK3.GetString(SDK3.AT_HANDLE_SYSTEM, 'SoftwareVersion', 255)


from PYME.Acquire.Hardware.Camera import Camera
class SDK3Camera(Camera):
    def __init__(self, camNum):
        """camera initialisation - note that this should be called  from derived classes
        *AFTER* the properties have been defined"""
        #camReg.regCamera() #initialise the library if needed
        self.camNum = camNum
        
        
    def Init(self):
        #print('Foo')
        self.handle = SDK3.Open(self.camNum)
        self.connectProperties()
        
    
    def connectProperties(self):
        for name, var in self.__dict__.items():
            if isinstance(var, ATProperty):
                var.connect(self.handle, name)
                
        
    def shutdown(self):
        SDK3.Close(self.handle)
        #camReg.unregCamera()
        
    
        
        
        
    
