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

#from PYME.Acquire.Hardware.AndorIXon import AndorIXon
from PYME.Acquire.Hardware.AndorNeo import AndorZyla
from PYME.Acquire.Hardware.AndorNeo import ZylaControlPanel
from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame

from PYME.Acquire.Hardware.Simulator import fakeCam, fakePiezo
from PYME.Acquire.Hardware import fakeShutters
import time
import os
import sys
import scipy

def GetComputerName():
    if sys.platform == 'win32':
        return os.environ['COMPUTERNAME']
    else:
        return os.uname()[1]

#scope.cameras = {}
#scope.camControls = {}
from PYME.IO import MetaDataHandler

InitBG('Andor Zyla', '''
scope.cameras['Zyla'] =  AndorZyla.AndorZyla(0)
scope.cam = scope.cameras['Zyla']
scope.cameras['Zyla'].Init()
scope.cameras['Zyla'].port = 'R100'
scope.cameras['Zyla'].orientation = dict(rotate=False, flipx=False, flipy=False)
''')

InitGUI('''
scope.camControls['Zyla'] = ZylaControlPanel.ZylaControl(MainFrame, scope.cameras['Zyla'], scope)
camPanels.append((scope.camControls['Zyla'], 'sCMOS Properties'))
''')



# InitGUI('''
# from PYME.Acquire import sampleInformation
# sampPan = sampleInformation.slidePanel(MainFrame)
# camPanels.append((sampPan, 'Current Slide'))
# ''')
InitGUI("""
from PYME.Acquire import sampleInformationDjangoDirect as sampleInformation
sampPan = sampleInformation.slidePanel(MainFrame)
MetaDataHandler.provideStartMetadata.append(lambda mdh: sampleInformation.getSampleDataFailesafe(MainFrame,mdh))
camPanels.append((sampPan, 'Current Slide'))
""")

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [fakeShutters.CH1] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo
scope.shutters = fakeShutters

InitGUI('''
from PYME.Acquire import sarcSpacing
ssp = sarcSpacing.SarcomereChecker(MainFrame, menuBar1, scope)
''')




#Z stage
#InitGUI('''
#from PYME.Acquire.Hardware import NikonTi
#scope.zStage = NikonTi.zDrive()
#scope.piezos.append((scope.zStage, 1, 'Z Stepper'))
#''')# % GetComputerName())

InitBG('Z Piezo', '''
from PYME.Acquire.Hardware.Piezos import piezo_e709, offsetPiezo

scope._piFoc = piezo_e709.piezo_e709T('COM12', 400, 0, True)
scope.hardwareChecks.append(scope._piFoc.OnTarget)
scope.CleanupFunctions.append(scope._piFoc.close)
scope.piFoc = offsetPiezo.piezoOffsetProxy(scope._piFoc)
scope.piezos.append((scope.piFoc, 1, 'PIFoc'))
scope.positioning['z'] = (scope.piFoc, 1, 1)

#server so drift correction can connect to the piezo
scope.pst = offsetPiezo.ServerThread(scope.piFoc)
scope.pst.start()
scope.CleanupFunctions.append(scope.pst.cleanup)

scope.state.registerHandler('Positioning.z', lambda : scope.piFoc.GetPos(1), lambda v : scope.piFoc.MoveTo(1, v))
''')

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

InitGUI("""
from PYME.Acquire import positionTracker
pt = positionTracker.PositionTracker(scope, time1)
pv = positionTracker.TrackerPanel(MainFrame, pt)
MainFrame.AddPage(page=pv, select=False, caption='Track')
time1.WantNotification.append(pv.draw)
""")

#splitter
InitGUI("""
from PYME.Acquire.Hardware import splitter
splt = splitter.Splitter(MainFrame, scope, scope.cam, flipChan = 0, transLocOnCamera = 'Left', flip=False, dir='left_right', constrain=False)
""")

InitGUI('''
from PYME.Acquire.Hardware import focusKeys
fk = focusKeys.FocusKeys(MainFrame, scope.piezos[0], scope=scope)
time1.WantNotification.append(fk.refresh)
''')



#we don't have a splitter - make sure that the analysis knows this
#scope.mdh['Splitter.Flip'] = False


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
    
##from PYME.Acquire.Hardware.FilterWheel import WFilter, FiltFrame
##filtList = [WFilter(1, 'EMPTY', 'EMPTY', 0),
##    WFilter(2, 'ND.5' , 'UVND 0.5', 0.5),
##    WFilter(3, 'ND1'  , 'UVND 1'  , 1),
##    WFilter(4, 'ND2', 'UVND 2', 2),
##    WFilter(5, 'ND3'  , 'UVND 3'  , 3),
##    WFilter(6, 'ND4'  , 'UVND 4'  , 4)]
##
##InitGUI('''
##try:
##    scope.filterWheel = FiltFrame(MainFrame, filtList, 'COM4')
##    scope.filterWheel.SetFilterPos("ND4")
##    toolPanels.append((scope.filterWheel, 'Filter Wheel'))
##except:
##    print 'Error starting filter wheel ...'
##''')


#DigiData
##from PYME.Acquire.Hardware import phoxxLaser
##scope.l642 = phoxxLaser.PhoxxLaser('642')
##scope.StatusCallbacks.append(scope.l642.GetStatusText)
##scope.lasers = [scope.l642]
##scope.lasers = []

from PYME.Acquire.Hardware import lasers
sb = lasers.SBox(com_port='COM5')
scope.l671 = lasers.SerialSwitchedLaser('671',sb,0)
scope.l532 = lasers.SerialSwitchedLaser('532',sb,2)

from PYME.Acquire.Hardware import matchboxLaser
scope.l405 = matchboxLaser.MatchboxLaser('405',portname='COM4')

from PYME.Acquire.Hardware import phoxxLaserOLD
scope.l647 = phoxxLaserOLD.PhoxxLaser('647',portname='COM6')
scope.StatusCallbacks.append(scope.l647.GetStatusText)

scope.lasers = [scope.l671, scope.l405, scope.l647, scope.l532]


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
    camPanels.append((lcf, 'Laser Control'))
''')

from PYME.Acquire.Hardware.FilterWheel import WFilter, FiltFrame, FiltWheel
filtList = [WFilter(1, 'EMPTY', 'EMPTY', 0),
    WFilter(2, 'ND.5' , 'UVND 0.5', 0.5),
    WFilter(3, 'ND1'  , 'UVND 1'  , 1),
    WFilter(4, 'ND2', 'UVND 2', 2),
    WFilter(5, 'ND3'  , 'UVND 3'  , 3),
    WFilter(6, 'ND4'  , 'UVND 4'  , 4)]

InitGUI('''
try:
    scope.filterWheel = FiltWheel(filtList, 'COM7')
    #scope.filterWheel.SetFilterPos("LF488")
    scope.filtPan = FiltFrame(MainFrame, scope.filterWheel)
    toolPanels.append((scope.filtPan, 'Filter Wheel'))
except:
    print 'Error starting filter wheel ...'
''')

from PYME.Acquire.Hardware import ExciterWheel
exciterList = [ExciterWheel.WFilter(1, 'GFP', 'GFP exciter', 0),
    ExciterWheel.WFilter(2, 'TxRed' , 'TxRed exciter', 0),
    ExciterWheel.WFilter(3, 'Cy5'  , 'Cy5 exciter'  , 0),
    ExciterWheel.WFilter(4, 'Cy5.5', 'Cy5.5 exciter', 0),
    ExciterWheel.WFilter(5, 'Cy7'  , 'Cy7 exciter'  , 0),
    ExciterWheel.WFilter(6, 'ND4'  , 'ND4'  , 0)]

filterpair = [ExciterWheel.FilterPair('GFP', 'GFP'),
    ExciterWheel.FilterPair('TxRed', 'TxRed'),
    ExciterWheel.FilterPair('Cy5', 'Cy5'),
    ExciterWheel.FilterPair('Cy5.5', 'Cy5.5'),
    ExciterWheel.FilterPair('Cy7', 'Cy7'),
    ExciterWheel.FilterPair('To be added', 'To be added')]

InitGUI('''
try:
    scope.exciterWheel = ExciterWheel.FiltWheel(exciterList, filterpair, 'COM14', dichroic=scope.dichroic)
    #scope.filterWheel.SetFilterPos("LF488")
    scope.exciterPan = ExciterWheel.FiltFrame(MainFrame, scope.exciterWheel)
    toolPanels.append((scope.exciterPan, 'Exciter Wheel'))
except:
    print 'Error starting exciter wheel ...'
''')


InitBG('XY Stage', '''
#XY Stage
from PYME.Acquire.Hardware.Piezos import piezo_c867
scope.xystage = piezo_c867.piezo_c867T('COM13')
scope.piezos.append((scope.xystage, 2, 'Stage_X'))
scope.piezos.append((scope.xystage, 1, 'Stage_Y'))
scope.joystick = piezo_c867.c867Joystick(scope.xystage)
#scope.joystick.Enable(True)
scope.hardwareChecks.append(scope.xystage.OnTarget)
scope.CleanupFunctions.append(scope.xystage.close)

scope.positioning['x'] = (scope.xystage, 1, 1000)
scope.positioning['y'] = (scope.xystage, 2, -1000)

scope.state.registerHandler('Positioning.x', lambda : 1000*scope.xystage.GetPos(1), lambda v : scope.xystage.MoveTo(1, v*1e-3))
scope.state.registerHandler('Positioning.y', lambda : -1000*scope.xystage.GetPos(2), lambda v : scope.xystage.MoveTo(2, -v*1e-3))
''')

InitGUI('''
from PYME.Acquire.Hardware import focusKeys
Posk = focusKeys.PositionKeys(MainFrame, menuBar1, scope.piezos[1], scope.piezos[2], scope=scope)
#time1.WantNotification.append(fk.refresh)
''')

InitGUI('''
from PYME.Acquire.Hardware import spacenav
scope.spacenav = spacenav.SpaceNavigator()
scope.CleanupFunctions.append(scope.spacenav.close)
scope.ctrl3d = spacenav.SpaceNavPiezoCtrl(scope.spacenav, scope.piFoc, scope.xystage)
''')

from PYME.Acquire.Hardware import priorLumen, arclampshutterpanel
InitGUI('''
try:
    scope.arclampshutter = priorLumen.PriorLumen('Arc lamp shutter', portname='COM23')
    scope.shuttercontrol = [scope.arclampshutter]
    acf = arclampshutterpanel.Arclampshutterpanel(MainFrame,scope.shuttercontrol)
    time1.WantNotification.append(acf.refresh)
    camPanels.append((acf, 'Shutter Control'))
except:
    print 'Error starting arc-lamp shutter ...'
''')

InitGUI("""
from PYME.Acquire.ui import actionUI

ap = actionUI.ActionPanel(MainFrame, scope.actions, scope)
MainFrame.AddPage(ap, caption='Queued Actions')
""")

InitGUI("""
from PYME.Acquire.ui import AnalysisSettingsUI
AnalysisSettingsUI.Plug(scope, MainFrame)
""")

# InitBG('DMD', '''
# from PYME.Acquire.Hardware import TiLightCrafter

# scope.LC = TiLightCrafter.LightCrafter()
# scope.LC.Connect()
# scope.LC.SetDisplayMode(scope.LC.DISPLAY_MODE.DISP_MODE_IMAGE)
# scope.LC.SetStatic(255)
# ''')

# InitGUI('''
# from PYME.Acquire.Hardware import DMDGui
# DMDModeSelectionPanel = DMDGui.DMDModeChooserPanel(MainFrame, scope)
# DMDtpPanel = DMDGui.DMDTestPattern(MainFrame, scope.LC)
# DMDsiPanel = DMDGui.DMDStaticImage(MainFrame, scope.LC)
# DMDseqPanel = DMDGui.DMDImageSeq(MainFrame, scope.LC)
# camPanels.append((DMDModeSelectionPanel, 'select DMD Mode'))
# camPanels.append((DMDtpPanel, 'select test pattern'))
# camPanels.append((DMDsiPanel, 'select static image'))
# camPanels.append((DMDseqPanel, 'select image sequence'))
# ''')

#from PYME.Acquire.Hardware import priorLumen
#scope.arclamp = priorLumen.PriorLumen('Arc Lamp', portname='COM6')
#scope.lasers.append(scope.arclamp)

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


#
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

# from PYME import cSMI


# Is = []

# def calcSum(caller):
#     Is.append(cSMI.CDataStack_AsArray(caller.ds, 0).sum())


#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#scope.SetCamera('A')

time.sleep(.5)
scope.initDone = True
