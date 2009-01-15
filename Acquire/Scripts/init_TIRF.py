from Hardware.AndorIXon import AndorIXon
from Hardware.AndorIXon import AndorControlFrame

from Hardware import fakeShutters

#from PYME import cSMI

scope.cam = AndorIXon.iXonCamera()

acf = AndorControlFrame.AndorFrame(MainFrame, scope.cam, scope)
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
frs = HDFSpoolFrame.FrSpool(MainFrame, scope, 'd:\\%(username)s\\%(day)d-%(month)d-%(year)d\\')
frs.Show()


#Z stage
from Hardware import NikonTE2000
scope.zStage = NikonTE2000.zDrive()
scope.piezos.append((scope.zStage, 1, 'Z Stepper'))

from Hardware import frZStage
frz = frZStage.frZStepper(MainFrame, scope.zStage)
frz.Show()

#3-axis piezo
from Hardware import thorlabsPiezo

#check to see what we've got attached
piezoSerialNums = thorlabsPiezo.EnumeratePiezos()
if len(piezoSerialNums) == 3: #expect to see 3 piezos
    scope.pzx = thorlabsPiezo.TLPiezo(91814461, 'X Axis')
    scope.pzy = thorlabsPiezo.TLPiezo(91814462, 'Y Axis')
    scope.pzz = thorlabsPiezo.TLPiezo(91814463, 'Z Axis')

    scope.piezos.append((scope.pzx, 1, 'X Piezo'))
    scope.piezos.append((scope.pzy, 1, 'Y Piezo'))
    scope.piezos.append((scope.pzz, 1, 'Z Piezo'))

    #centre the piezos
    scope.pzx.MoveTo(0,50)
    scope.pzy.MoveTo(0,50)
    scope.pzz.MoveTo(0,40)
    
from Hardware.FilterWheel import WFilter, FiltFrame
filtList = [WFilter(1, 'EMPTY', 'EMPTY', 0),
    WFilter(2, 'ND.5' , 'UVND 0.5', 0.5),
    WFilter(3, 'ND1'  , 'UVND 1'  , 1),
    WFilter(4, 'ND1.5', 'UVND 1.5', 1.5),
    WFilter(5, 'ND2'  , 'UVND 2'  , 2),
    WFilter(6, 'ND3'  , 'UVND 3'  , 3)]

scope.filterWheel = FiltFrame(MainFrame, filtList)
scope.filterWheel.Show()

#DigiData
from Hardware.DigiData import DigiDataClient
dd = DigiDataClient.getDDClient()

from Hardware import lasers
l488 = lasers.DigiDataSwitchedLaser('488',dd,1)
l405 = lasers.DigiDataSwitchedLaserInvPol('405',dd,0)
l473 = lasers.DigiDataSwitchedAnalogLaser('473',dd,0)

scope.lasers = [l488,l405,l473]

from Hardware import LaserControlFrame
lcf = LaserControlFrame.LaserControl(MainFrame,scope.lasers)
lcf.Show()

from Hardware import FocCorrR
fc = FocCorrR.FocusCorrector(scope.zStage, tolerance=0.20000000000000001, estSlopeDyn=False, recDrift=False, axis='Y', guideLaser=l488)
scope.StatusCallbacks.append(fc.GetStatus)
fc.addMenuItems(MainFrame, MainMenu)
fc.Start(2000)

from PYME import cSMI
import time

Is = []

def calcSum(caller):
    Is.append(cSMI.CDataStack_AsArray(caller.ds, 0).sum())

#scope.pa.WantFrameNotification.append(calcSum)
