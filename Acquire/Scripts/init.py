#!/usr/bin/python

import scipy
from Hardware.Simulator import fakeCam, fakePiezo, lasersliders, dSimControl
from Hardware import fakeShutters
import time

#import PYME.cSMI as example

pz = InitBG('Fake Piezo', '''
scope.fakePiezo = fakePiezo.FakePiezo(100)
scope.piezos.append((scope.fakePiezo, 1, 'Fake z-piezo'))
#time.sleep(5)
''')

pz.join() #piezo must be there before we start camera
cm = InitBG('Fake Camera', '''
scope.cam = fakeCam.FakeCamera(70*scipy.arange(-128.0, 128.0), 70*scipy.arange(-128.0, 128.0), fakeCam.NoiseMaker(), scope.fakePiezo)
#time.sleep(5)
''')

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [0] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo

scope.shutters = fakeShutters

#InitBG('Should Fail', '''
#raise Exception, 'test error'
#time.sleep(1)
#''')
#
#InitBG('Should not be there', '''
#raise HWNotPresent, 'test error'
#time.sleep(1)
#''')


#Gui stuff can't be done in background
InitGUI('''
dsc = dSimControl.dSimControl(notebook1, scope)
notebook1.AddPage(page=dsc, select=False, caption='Simulation Settings')
''')

InitGUI('''
from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame
acf = AndorControlFrame.AndorPanel(MainFrame, scope.cam, scope)
camPanels.append((acf, 'EMCCD Properties'))
''')

InitGUI('''
from PYME.Acquire.Hardware import ccdAdjPanel
#import wx
#f = wx.Frame(None)
snrPan = ccdAdjPanel.sizedCCDPanel(notebook1, scope, acf)
notebook1.AddPage(page=snrPan, select=False, caption='Image SNR')
#camPanels.append((snrPan, 'SNR etc ...'))
#f.Show()
#time1.WantNotification.append(snrPan.ccdPan.draw)
''')

cm.join()
from PYME.Acquire.Hardware import lasers
scope.l488 = lasers.FakeLaser('488',scope.cam,1, initPower=50)
scope.l405 = lasers.FakeLaser('405',scope.cam,0, initPower=10)

scope.lasers = [scope.l405, scope.l488]

InitGUI('''
from PYME.Acquire.Hardware import LaserControlFrame
lcf = LaserControlFrame.LaserControlLight(MainFrame,scope.lasers)
time1.WantNotification.append(lcf.refresh)
#lcf.Show()
toolPanels.append((lcf, 'Laser Control'))
''')

InitGUI('''
lsf = lasersliders.LaserSliders(toolPanel, scope.lasers)
toolPanels.append((lsf, 'Laser Powers'))
''')


InitGUI('''
from PYME.Acquire.Hardware import focusKeys
fk = focusKeys.FocusKeys(MainFrame, menuBar1, scope.piezos[-1])
''')

from PYME import cSMI

Is = []

def calcSum(caller):
    Is.append(cSMI.CDataStack_AsArray(caller.ds, 0).sum())

#scope.pa.WantFrameNotification.append(calcSum)

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

import numpy
psf = numpy.load('/home/david/psf488.npy')
psf = numpy.maximum(psf, 0.)
from PYME.Analysis import MetaData
fakeCam.rend_im.setModel(psf, MetaData.TIRFDefault)

time.sleep(.5)
scope.initDone = True
