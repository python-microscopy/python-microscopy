from Hardware.AndorIXon import AndorIXon
from Hardware.AndorIXon import AndorControlFrame

from Hardware import fakeShutters

#from PYME import cSMI

scope.cam = AndorIXon.iXonCamera()

acf = AndorControlFrame.AndorFrame(None, scope.cam, scope)
acf.Show()



#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [fakeShutters.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo
scope.shutters = fakeShutters

import HDFSpoolFrame
frs = HDFSpoolFrame.FrSpool(None, scope, 'd:\\%(username)s\\%(day)d-%(month)d-%(year)d\\')
frs.Show()


#Z stage
from Hardware import NikonTE2000
scope.zStage = NikonTE2000.zDrive()
scope.piezos.append((scope.zStage, 1, 'Z Stepper'))

from Hardware import frZStage
frz = frZStage.frZStepper(None, scope.zStage)
frz.Show()


#DigiData
from Hardware.DigiData import DigiDataClient
dd = DigiDataClient.getDDClient()

from Hardware import lasers
l488 = lasers.DigiDataSwitchedLaser('488',dd,1)
l405 = lasers.DigiDataSwitchedLaserInvPol('405',dd,0)
l473 = lasers.DigiDataSwitchedAnalogLaser('473',dd,0)

scope.lasers = [l488,l405,l473]

from Hardware import LaserControlFrame
lcf = LaserControlFrame.LaserControl(scope.lasers)
lcf.Show()

from PYME import cSMI
import time

Is = []

def calcSum(caller):
    Is.append(cSMI.CDataStack_AsArray(caller.ds, 0).sum())

scope.pa.WantFrameNotification.append(calcSum)
