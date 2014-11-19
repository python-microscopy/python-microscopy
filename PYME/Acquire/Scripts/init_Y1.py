#!/usr/bin/python

##################
# init_TIRF.py
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

from PYME.Acquire.Hardware.AndorIXon import AndorIXon
from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame
from PYME.Acquire.Hardware.uc480 import uCam480

from PYME.Acquire.Hardware import fakeShutters
import time
import os
import sys

def GetComputerName():
    if sys.platform == 'win32':
        return os.environ['COMPUTERNAME']
    else:
        return os.uname()[1]

#scope.cameras = {}
#scope.camControls = {}
from PYME.Acquire import MetaDataHandler

InitBG('EMCCD Cameras', '''
scope.cameras['A - Left'] = AndorIXon.iXonCamera(0)
scope.cameras['B - Right'] = uCam480.uc480Camera(1)
scope.cameras['A - Left'].port = 'L100'
scope.cameras['B - Right'].port = 'R100'
#scope.cameras['B - Right'].SetShutter(False)
scope.cameras['B - Right'].SetActive(False)
scope.cam = scope.cameras['A - Left']

''')

#InitBG('EMCCD Camera 2', '''
#scope.cameras['B'] = AndorIXon.iXonCamera(0)
#''')

InitGUI('''
scope.camControls['A - Left'] = AndorControlFrame.AndorPanel(MainFrame, scope.cameras['A - Left'], scope)
camPanels.append((scope.camControls['A - Left'], 'EMCCD A Properties'))

scope.camControls['B - Right'] = AndorControlFrame.AndorPanel(MainFrame, scope.cameras['A - Left'], scope)
camPanels.append((scope.camControls['B - Right'], 'EMCCD B Properties'))

''')

InitGUI('''
from PYME.Acquire import sampleInformation
sampPan = sampleInformation.slidePanel(MainFrame)
camPanels.append((sampPan, 'Current Slide'))
''')

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [fakeShutters.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo
scope.shutters = fakeShutters



#Light crafter
InitGUI('''
from PYME.Acquire.Hardware import TiLightCrafter
from PYMEnf.Hardware import DMDGui
scope.LC = TiLightCrafter.LightCrafter()
scope.LC.Connect()
scope.LC.SetDisplayMode(scope.LC.DISPLAY_MODE.DISP_MODE_IMAGE)
scope.LC.SetStatic(255)

LCGui = DMDGui.DMDPanel(MainFrame,scope.LC, scope)
camPanels.append((LCGui, 'DMD Control'))
''')

#PIFoc
InitBG('Piezos', '''
from PYME.Acquire.Hardware.Piezos import piezo_e709, offsetPiezo
#from PYME.Acquire.Hardware import NikonTi                          ###
#scope.zStage = NikonTi.zDrive()
#scope.piezos.append((scope.zStage, 1, 'Z Stepper'))

scope._piFoc = piezo_e709.piezo_e709T('COM9', 400, 0, True)
scope.hardwareChecks.append(scope._piFoc.OnTarget)
scope.piFoc = offsetPiezo.piezoOffsetProxy(scope._piFoc)
scope.piezos.append((scope.piFoc, 1, 'PIFoc'))
scope.positioning['z'] = (scope.piFoc, 1, 1)

#server so drift correction can connect to the piezo
pst = offsetPiezo.ServerThread(scope.piFoc)
pst.start()


#XY Stage
from PYME.Acquire.Hardware.Piezos import piezo_c867
scope.xystage = piezo_c867.piezo_c867T('COM8')
scope.piezos.append((scope.xystage, 2, 'Stage_X'))
scope.piezos.append((scope.xystage, 1, 'Stage_Y'))
scope.joystick = piezo_c867.c867Joystick(scope.xystage)
#scope.joystick.Enable(True)
scope.hardwareChecks.append(scope.xystage.OnTarget)

scope.positioning['x'] = (scope.xystage, 1, 1000)
scope.positioning['y'] = (scope.xystage, 2, -1000)
''')

#InitBG('Stage Stepper Motors', '''
#from PYME.Acquire.Hardware.Mercury import mercuryStepper
#scope.stage = mercuryStepper.mercuryStepper(comPort=5, axes=['A', 'B'], steppers=['M-229.25S', 'M-229.25S'])
#scope.stage.SetSoftLimits(0, [1.06, 20.7])
#scope.stage.SetSoftLimits(1, [.8, 17.6])
#scope.piezos.append((scope.stage, 0, 'Stage X'))
#scope.piezos.append((scope.stage, 1, 'Stage Y'))
#scope.joystick = scope.stage.joystick
#scope.joystick.Enable(True)
#scope.CleanupFunctions.append(scope.stage.Cleanup)
#''')

#InitGUI('''
#from PYME.Acquire import sarcSpacing
#ssp = sarcSpacing.SarcomereChecker(MainFrame, menuBar1, scope)
#''')



InitGUI('''
from PYME.Acquire import positionTracker
pt = positionTracker.PositionTracker(scope, time1)
pv = positionTracker.TrackerPanel(MainFrame, pt)
MainFrame.AddPage(page=pv, select=False, caption='Track')
time1.WantNotification.append(pv.draw)
''')

#splitter
InitGUI('''
from PYME.Acquire.Hardware import splitter
splt = splitter.Splitter(MainFrame, mControls, scope, scope.cam, flipChan = 0, dichroic = 'FF700-Di01' , transLocOnCamera = 'Left', flip=False, dir='left_right', constrain=False)
''')

#we don't have a splitter - make sure that the analysis knows this
#scope.mdh['Splitter.Flip'] = False

#Nikon Ti motorised controls
InitGUI('''
from PYME.Acquire.Hardware import NikonTi, NikonTiGUI
scope.dichroic = NikonTi.FilterChanger()
scope.lightpath = NikonTi.LightPath()

TiPanel = NikonTiGUI.TiPanel(MainFrame, scope.dichroic, scope.lightpath)
toolPanels.append((TiPanel, 'Nikon Ti'))
#time1.WantNotification.append(TiPanel.SetSelections)
time1.WantNotification.append(scope.dichroic.Poll)
time1.WantNotification.append(scope.lightpath.Poll)

MetaDataHandler.provideStartMetadata.append(scope.dichroic.ProvideMetadata)
MetaDataHandler.provideStartMetadata.append(scope.lightpath.ProvideMetadata)
''')# % GetComputerName())

#InitGUI('''
#from PYME.Acquire.Hardware import focusKeys
#fk = focusKeys.FocusKeys(MainFrame, menuBar1, scope.piezos[0], scope=scope)
#time1.WantNotification.append(fk.refresh)
#
#xykeys = focusKeys.PositionKeys(MainFrame, menuBar1, scope.piezos[1], scope.piezos[2], scope=scope)
#''')

InitGUI('''
from PYME.Acquire.Hardware import spacenav
scope.spacenav = spacenav.SpaceNavigator()
scope.ctrl3d = spacenav.SpaceNavPiezoCtrl(scope.spacenav, scope.piFoc, scope.xystage)
''')

#from PYME.Acquire.Hardware import frZStage
#frz = frZStage.frZStepper(MainFrame, scope.zStage)
#frz.Show()

##3-axis piezo
#InitBG('Thorlabs Piezo', '''
#from PYME.Acquire.Hardware import thorlabsPiezo
#
##check to see what we've got attached
#piezoSerialNums = thorlabsPiezo.EnumeratePiezos()
#if len(piezoSerialNums) == 3: #expect to see 3 piezos
#    scope.pzx = thorlabsPiezo.TLPiezo(91814461, 'X Axis')
#    scope.pzy = thorlabsPiezo.TLPiezo(91814462, 'Y Axis')
#    scope.pzz = thorlabsPiezo.TLPiezo(91814463, 'Z Axis')
#
#    scope.piezos.append((scope.pzx, 1, 'X Piezo'))
#    scope.piezos.append((scope.pzy, 1, 'Y Piezo'))
#    scope.piezos.append((scope.pzz, 1, 'Z Piezo'))
#
#    #centre the piezos
#    scope.pzx.MoveTo(0,50)
#    scope.pzy.MoveTo(0,50)
#    scope.pzz.MoveTo(0,40)
#else:
#    raise HWNotPresent
#
#''')
    
from PYME.Acquire.Hardware.FilterWheel import WFilter, FiltFrame
filtList = [WFilter(1, 'LF405', 'LF405', 0),
    WFilter(2, 'LF488' , 'LF488', 0),
    WFilter(3, 'LF561'  , 'LF561'  , 0),
    WFilter(4, '647LP', '647LP', 0),
    WFilter(5, 'Cy7'  , 'Cy7'  , 0),
    WFilter(6, 'EMPTY'  , 'EMPTY'  , 0)]

InitGUI('''
try:
    scope.filterWheel = FiltFrame(MainFrame, filtList, 'COM11', dichroic=scope.dichroic)
    scope.filterWheel.SetFilterPos("LF488")
    toolPanels.append((scope.filterWheel, 'Filter Wheel'))
except:
    print 'Error starting filter wheel ...'
''')


#DigiData
from PYME.Acquire.Hardware import phoxxLaser, cobaltLaser, ioslave
scope.l642 = phoxxLaser.PhoxxLaser('642',portname='COM4')
scope.CleanupFunctions.append(scope.l642.Close)
scope.l488 = phoxxLaser.PhoxxLaser('488',portname='COM5')
scope.CleanupFunctions.append(scope.l488.Close)
scope.l405 = phoxxLaser.PhoxxLaser('405',portname='COM6')
scope.CleanupFunctions.append(scope.l405.Close)
scope.l561 = cobaltLaser.CobaltLaser('561',portname='COM7')
scope.lAOM = ioslave.AOMLaser('AOM')
scope.lasers = [scope.l405,scope.l488,scope.l561, scope.lAOM, scope.l642]

#scope.lasers = [scope.l405,scope.l488, scope.l642]
#scope.lasers = [scope.l642]

#scope.StatusCallbacks.append(scope.l642.GetStatusText)
#scope.lasers = [scope.l642]
#scope.lasers = []
#InitBG('DigiData', '''
#from PYME.Acquire.Hardware.DigiData import DigiDataClient
#dd = DigiDataClient.getDDClient()
#
#
#from PYME.Acquire.Hardware import lasers
#scope.l490 = lasers.DigiDataSwitchedLaser('490',dd,4)
#scope.l405 = lasers.DigiDataSwitchedLaserInvPol('405',dd,0)
##scope.l543 = lasers.DigiDataSwitchedAnalogLaser('543',dd,0)
##scope.l671 = lasers.DigiDataSwitchedAnalogLaser('671',dd,1)
#
#pport = lasers.PPort()
#scope.l671 = lasers.ParallelSwitchedLaser('671',pport,0)
#scope.l532 = lasers.ParallelSwitchedLaser('532',pport,1)
#
#scope.lasers = [scope.l405,scope.l532,scope.l671, scope.l490]
#''')

from PYME.Acquire.Hardware import priorLumen
scope.arclamp = priorLumen.PriorLumen('Arc Lamp', portname='COM1')
scope.lasers.append(scope.arclamp)

#scope.lasers = [scope.arclamp]

#scope.lasers = []

InitGUI('''
from PYME.Acquire import lasersliders
lsf = lasersliders.LaserSliders(toolPanel, scope.lasers)
time1.WantNotification.append(lsf.update)
#lsf.update()
camPanels.append((lsf, 'Laser Powers'))
''')

InitGUI('''
if 'lasers'in dir(scope):
    from PYME.Acquire.Hardware import LaserControlFrame
    lcf = LaserControlFrame.LaserControlLight(MainFrame,scope.lasers)
    time1.WantNotification.append(lcf.refresh)
    #lcf.refresh()
    camPanels.append((lcf, 'Laser Control'))
''')

#from PYME.Acquire.Hardware import PM100USB
#
#scope.powerMeter = PM100USB.PowerMeter()
#scope.powerMeter.SetWavelength(671)
#scope.StatusCallbacks.append(scope.powerMeter.GetStatusText)

##Focus tracking
#from PYME.Acquire.Hardware import FocCorrR
#InitBG('Focus Corrector', '''
#scope.fc = FocCorrR.FocusCorrector(scope.zStage, tolerance=0.20000000000000001, estSlopeDyn=False, recDrift=False, axis='Y', guideLaser=l488)
#scope.StatusCallbacks.append(fc.GetStatus)
#''')
#InitGUI('''
#if 'fc' in dir(scope):
#    scope.fc.addMenuItems(MainFrame, MainMenu)
#    scope.fc.Start(2000)
#''')

from PYME import cSMI


Is = []

def calcSum(caller):
    Is.append(cSMI.CDataStack_AsArray(caller.ds, 0).sum())


#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#scope.SetCamera('A')

time.sleep(.5)
scope.initDone = True
