#!/usr/bin/python

##################
# lasers.py
#
# Copyright David Baddeley, 2009
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
##################

import time
import threading

##Virtual laser class - to be overridden

class Laser(object):
    powerControlable = False
    MAX_POWER = 1
    MIN_POWER = 0
    units='\%'

    def __init__(self, name,turnOn=False, scopeState=None):
        self.name= name
        if turnOn:
            self.TurnOn()
        else:
            self.TurnOff()
            
        if scopeState:
            self.registerStateHandlers(scopeState)

    def IsOn(self):
        return False

    def TurnOn(self):
        pass

    def TurnOff(self):
        pass

    def SetPower(self, power):
        pass

    def GetPower(self):
        return -1

    def Pulse(self, t):
        self.TurnOn()
        time.sleep(t)
        self.TurnOff()

    def PulseBG(self, t): #do pulse in background to avoid blocking main program
        th = threading.Thread(target = self.Pulse, args=(t,))
        th.start()

    def IsPowerControlable(self):
        return self.powerControlable

    def GetName(self):
        return self.name
    
    def register(self, scope):
        scope.lasers.append(self)
        self.registerStateHandlers(scope.state)
        
    def registerStateHandlers(self, scopeState):
        scopeState.registerHandler('Lasers.%s.On' % self.name, self.IsOn, lambda v: self.TurnOn() if v else self.TurnOff())
        if self.IsPowerControlable():
            scopeState.registerHandler('Lasers.%s.Power' % self.name, self.GetPower, self.SetPower)
            scopeState.registerHandler('Lasers.%s.MaxPower' % self.name, lambda : self.MAX_POWER)
            scopeState.registerHandler('Lasers.%s.MinPower' % self.name, lambda : self.MIN_POWER)



#laser which is attached to a DigiData digital io channel
class DigiDataSwitchedLaser(Laser):
    def __init__(self, name,digiData, digIOChan, **kwargs):
        self.dd = digiData
        self.chan = digIOChan  

        Laser.__init__(self,name, **kwargs)

    def IsOn(self):
        return self.dd.GetDOBit(self.chan)

    def TurnOn(self):
        self.dd.SetDOBit(self.chan)

    def TurnOff(self):
        self.dd.UnsetDOBit(self.chan)


#laser which is attached to a DigiData digital io channel, with negative polarity (0 = on, 1 = off) 
class DigiDataSwitchedLaserInvPol(Laser):
    def __init__(self,name, digiData, digIOChan, **kwargs):
        self.dd = digiData
        self.chan = digIOChan  

        Laser.__init__(self,name, **kwargs)

    def IsOn(self):
        return not self.dd.GetDOBit(self.chan)

    def TurnOn(self):
        self.dd.UnsetDOBit(self.chan)

    def TurnOff(self):
        self.dd.SetDOBit(self.chan)


#laser which is attached to a DigiData analog io channel
class DigiDataSwitchedAnalogLaser(Laser):
    def __init__(self,name, digiData, AOChan, initPower=1, fullScaleAOVal = 2**14, offAOVal = 0 , **kwargs):
        self.dd = digiData
        self.chan = AOChan
        self.power = initPower

        #Laser needs voltage between 0 and 10V - analog output is 16bit signed integer - > 2**14 = 10V
        self.fullScaleAOVal = fullScaleAOVal 
        self.offAOVal = offAOVal

        self.powerControlable = True
        Laser.__init__(self,name, **kwargs)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self.dd.PutAOValue(self.chan, int(self.power*(self.fullScaleAOVal - self.offAOVal)))
        self.isOn = True

    def TurnOff(self):
        self.dd.PutAOValue(self.chan, self.offAOVal)
        self.isOn = False

    def SetPower(self, power):
        if power < 0 or power > 1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and 1')
        self.power = power
        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def GetPower(self):
        return self.power


#laser which is attached to a DigiData analog io channel
class FakeLaser(Laser):
    def __init__(self,name, cam, chan, turnOn=False, initPower=1, maxPower=1e3, **kwargs):
        self.cam = cam
        self.chan = chan
        self.power = initPower

        
        
        self.MAX_POWER = maxPower

        self.powerControlable = True
        self.isOn = turnOn
        
        Laser.__init__(self,name,turnOn, **kwargs)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self.cam.compT.laserPowers[self.chan] = self.power
        self.cam.laserPowers[self.chan] = self.power
        self.isOn = True

    def TurnOff(self):
        self.cam.compT.laserPowers[self.chan] = 0
        self.cam.laserPowers[self.chan] = 0
        self.isOn = False

    def SetPower(self, power):
        self.power = power
        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def GetPower(self):
        return self.power


#port to allow bit-banging on parallel port
class PPort:
    def __init__(self,port=0, initVal = 0):
        import parallel
        self.port = parallel.Parallel(port)
        self.val = initVal
        self.port.setData(self.val)

    def setPin(self, pinNo, on):
        if on:
            self.val = self.val | 2**pinNo
        else:
            self.val = self.val & ~(2**pinNo)
            
        self.port.setData(self.val)

    def getPin(self, pinNo):
        return self.val & 2**pinNo



#laser which is attached to a DigiData digital io channel
class ParallelSwitchedLaser(Laser):
    def __init__(self, name, pport, pinNo, **kwargs):
        self.pport = pport
        self.pinNo = pinNo  

        Laser.__init__(self,name, **kwargs)

    def IsOn(self):
        return self.pport.getPin(self.pinNo)

    def TurnOn(self):
        self.pport.setPin(self.pinNo, True)

    def TurnOff(self):
        self.pport.setPin(self.pinNo, False)


# simple arduino-based serial box (made by Louey)
# maps several arduino pins to 12V outputs 0-2 on box
# arduino uses IOslave implementation, original from David Baddeley,
# I made a number of modifications
# serial protocol:
# 
class SBox:
    def __init__(self, com_port='COM4'):
        import serial
        self.ser =  serial.Serial(com_port, baudrate=19200, timeout=1)
        self.ioPorts = [0,0,0]
        self.ioMap = ['SD3','SD4','SD5']
        self.states = [False,False,False]
        for port in range(0,3):
            self.setState(port,False)
        self.status = 'ok'

    def setState(self, ioNo, on):
        if on:
            self.ser.write(self.ioMap[ioNo]+' 1\n')
            echo = self.ser.readline()
            self.status = self.ser.readline()
            if not self._checkstatusok():
                raise RuntimeError('Got invalid response %s' % (self.status))
            self.states[ioNo] = True
        else:
            self.ser.write(self.ioMap[ioNo]+' 0\n')
            echo = self.ser.readline()
            self.status = self.ser.readline()
            if not self._checkstatusok():
                raise RuntimeError('Got invalid response %s' % (self.status))
            self.states[ioNo] = False

    def getState(self, ioNo):
        return self.states[ioNo]

    def _checkstatusok(self):
        reps = self.status.split('\r')
        return reps[0] == 'ok'


class SerialSwitchedLaser(Laser):
    def __init__(self, name, sbox, ioNo, turnOn=False, **kwargs):
        self.sbox = sbox
        self.ioNo = ioNo

        Laser.__init__(self,name,turnOn, **kwargs)

    def IsOn(self):
        return self.sbox.getState(self.ioNo)

    def TurnOn(self):
        self.sbox.setState(self.ioNo, True)

    def TurnOff(self):
        self.sbox.setState(self.ioNo, False)
