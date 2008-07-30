#import sys
#sys.path.append(r'c:\pysmi\py_fit')

#import scipy
#import fakeCam
#import fakeCam
#reload(fakeCam)

from Hardware import AndorIXon
from Hardware.AndorIXon import AndorControlFrame

#import example
from PYME import cSMI

from Hardware import NikonTE2000
#example.CDataStack=fakeCam.CDataStack

#scope.fakePiezo = fakeCam.FakePiezo(100)
#scope.piezos.append((scope.fakePiezo, 1, 'Fake z-piezo'))

scope.zStage = NikonTE2000.zDrive()
scope.piezos.append((scope.zStage, 1, 'Z Stepper'))

#scope.cam = fakeCam.FakeCamera(70*scipy.arange(-128.0, 128.0), 70*scipy.arange(-128.0, 128.0), fakeCam.NoiseMaker(), scope.fakePiezo, fluors=[])

scope.cam = AndorIXon.iXonCamera()

acf = AndorControlFrame.AndorFrame(None, scope.cam, scope)
acf.Show()
#import vfr
#import previewaquisator
#import simplesequenceaquisator
#import funcs
#import piezo_e662
#import piezo_e816
#import psliders
#import intsliders
#import seqdialog

#scope = funcs.microscope()

#Setup for binning = 2 
#scope.cam.SetHorizBin(1)
#scope.cam.SetVertBin(1)
#scope.cam.SetCOC()
#scope.cam.GetStatus()

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [example.CShutterControl.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo

#setup the piezos
#scope.pfoc = piezo_e816.piezo_e816(1,100)
#scope.pfoc.MoveTo(1,50)

#scope.piezos.append((scope.pfoc, 1, 'Pfoc'))
#import lasersliders
#lsf = lasersliders.LaserSliders(scope.cam)
#lsf.Show()

#import dSimControl
#dsc = dSimControl.dSimControl(None, scope)
#dsc.Show()

#import remFit6
#import Pyro
#tq = Pyro.core.getProxyForURI('PYRONAME://taskQueue')


#def postTask(hi=None):
#    im = example.CDataStack_AsArray(scope.pa.ds,0)[:,:,0]
#    t = remFit6.fitTask(im, 10)
#    tq.postTask(t)

#scope.pa.WantFrameNotification.append(postTask)

#import remFitPSF
#import Pyro.core
#tq = Pyro.core.getProxyForURI('PYROLOC://130.216.133.27/taskQueue')

#import MetaData

#md = MetaData.MetaData(MetaData.VoxelSize(0.07,0.07,0.2))

#def postTask(hi=None):
#    im = example.CDataStack_AsArray(scope.pa.ds,0)[:,:,0]
#    t = remFitPSF.fitTask(im, 10,md)
#    tq.postTask(t)

#scope.pa.WantFrameNotification.append(postTask)

#import WebcamFrameCOI
#wcf = WebcamFrameCOI.WebcamFrame(None)
#wcf.Show()

from Hardware.DigiData import RemoteDigiData
dd = RemoteDigiData.getDDClient()

import HDFSpoolFrame
frs = HDFSpoolFrame.FrSpool(None, scope, 'd:\\%(username)s\\%(day)d-%(month)d-%(year)d\\')
frs.Show()

from Hardware import lasers
l488 = lasers.DigiDataSwitchedLaser('488',dd,1)
l405 = lasers.DigiDataSwitchedLaserInvPol('405',dd,0)
l473 = lasers.DigiDataSwitchedAnalogLaser('473',dd,0)

scope.lasers = [l488,l405,l473]

from Hardware import LaserControlFrame
lcf = LaserControlFrame.LaserControl(scope.lasers)
lcf.Show()

from Hardware import frZStage
frz = frZStage.frZStepper(None, scope.zStage)
frz.Show()

