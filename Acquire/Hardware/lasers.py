import time
import threading

##Virtual laser class - to be overridden

class Laser:
    powerControlable = False

    def __init__(self, name,turnOn=False):
        self.name= name
        if turnOn:
            self.TurnOn()
        else:
            self.TurnOff()

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


#laser which is attached to a DigiData digital io channel
class DigiDataSwitchedLaser(Laser):
    def __init__(self, name,digiData, digIOChan, turnOn=False):
        self.dd = digiData
        self.chan = digIOChan  

        Laser.__init__(self,name,turnOn)

    def IsOn(self):
        return self.dd.GetDOBit(self.chan)

    def TurnOn(self):
        self.dd.SetDOBit(self.chan)

    def TurnOff(self):
        self.dd.UnsetDOBit(self.chan)


#laser which is attached to a DigiData digital io channel, with negative polarity (0 = on, 1 = off) 
class DigiDataSwitchedLaserInvPol(Laser):
    def __init__(self,name, digiData, digIOChan, turnOn=False):
        self.dd = digiData
        self.chan = digIOChan  

        Laser.__init__(self,name,turnOn)

    def IsOn(self):
        return not self.dd.GetDOBit(self.chan)

    def TurnOn(self):
        self.dd.UnsetDOBit(self.chan)

    def TurnOff(self):
        self.dd.SetDOBit(self.chan)


#laser which is attached to a DigiData analog io channel
class DigiDataSwitchedAnalogLaser(Laser):
    def __init__(self,name, digiData, AOChan, turnOn=False, initPower=1, fullScaleAOVal = 2**14, offAOVal = 0):
        self.dd = digiData
        self.chan = AOChan
        self.power = initPower

        #Laser needs voltage between 0 and 10V - analog output is 16bit signed integer - > 2**14 = 10V
        self.fullScaleAOVal = fullScaleAOVal 
        self.offAOVal = offAOVal

        Laser.__init__(self,name,turnOn)

        self.powerControlable = True

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
            raise 'Error setting laser power: Power must be between 0 and 1'
        self.power = power
        if self.isOn:
            self.TurnOn() #turning on actually sets power

    def GetPower(self):
        return self.power


#laser which is attached to a DigiData analog io channel
class FakeLaser(Laser):
    def __init__(self,name, cam, chan, turnOn=False, initPower=1):
        self.cam = cam
        self.chan = chan
        self.power = initPower

        Laser.__init__(self,name,turnOn)

        self.powerControlable = True
        self.isOn = turnOn

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
