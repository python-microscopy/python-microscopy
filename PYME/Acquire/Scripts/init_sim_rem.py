#!/usr/bin/python

##################
# init.py
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

#!/usr/bin/python

import scipy
from PYME.Acquire.Hardware.Simulator import fakeCam, fakePiezo, lasersliders, dSimControl
from PYME.Acquire.Hardware import fakeShutters
from PYME.Acquire.Hardware.Piezos import offsetPiezo
import time

#import PYME.cSMI as example

pz = InitBG('Fake Piezo(s)', """
scope.fp = fakePiezo.FakePiezo(100)
scope.fakePiezo = offsetPiezo.piezoOffsetProxy(scope.fp)
scope.piezos.append((scope.fakePiezo, 1, 'Fake z-piezo'))

scope.fakeXPiezo = fakePiezo.FakePiezo(10)
scope.piezos.append((scope.fakeXPiezo, 1, 'Fake x-piezo'))

scope.fakeYPiezo = fakePiezo.FakePiezo(10)
scope.piezos.append((scope.fakeYPiezo, 1, 'Fake y-piezo'))
#time.sleep(5)
""")

pz.join() #piezo must be there before we start camera
cm = InitBG('Fake Camera', """
scope.cam = fakeCam.FakeCamera(70*scipy.arange(-128.0, 128.0), 70*scipy.arange(-128.0, 128.0), fakeCam.NoiseMaker(), scope.fp, xpiezo = scope.fakeXPiezo, ypiezo = scope.fakeYPiezo)
scope.cameras['Fake Camera'] = scope.cam
#time.sleep(5)
""")

#setup for the channels to aquire - b/w camera, no shutters
class chaninfo:
    names = ['bw']
    cols = [1] #1 = b/w, 2 = R, 4 = G1, 8 = G2, 16 = B
    hw = [0] #unimportant - as we have no shutters
    itimes = [100]

scope.chaninfo = chaninfo

scope.shutters = fakeShutters

#scope.EnableJoystick = 'foo'

#InitBG('Should Fail', """
#raise Exception, 'test error'
#time.sleep(1)
#""")
#
#InitBG('Should not be there', """
#raise HWNotPresent, 'test error'
#time.sleep(1)
#""")


#Gui stuff can't be done in background
InitGUI("""
dsc = dSimControl.dSimControl(MainFrame, scope)
#import wx
#dsc = wx.TextCtrl(MainFrame, -1, 'foo')
MainFrame.AddPage(page=dsc, select=False, caption='Simulation Settings')
""")

InitGUI("""
from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame
scope.camControls['Fake Camera'] = AndorControlFrame.AndorPanel(MainFrame, scope.cam, scope)
camPanels.append((scope.camControls['Fake Camera'], 'EMCCD Properties'))
""")

#InitGUI("""
#from PYME.Acquire import sampleInformation
#sampPan = sampleInformation.slidePanel(MainFrame)
#camPanels.append((sampPan, 'Current Slide'))
#""")

#InitGUI("""
#from PYMEnf.Hardware import FakeDMD, DMDGui
#scope.LC = FakeDMD.FakeDMD(scope)
#
#LCGui = DMDGui.DMDPanel(MainFrame,scope.LC, scope)
#camPanels.append((LCGui, 'DMD Control'))
#""")

#InitGUI("""
#from PYME.Acquire.Hardware import ccdAdjPanel
##import wx
##f = wx.Frame(None)
#snrPan = ccdAdjPanel.sizedCCDPanel(notebook1, scope, acf)
#notebook1.AddPage(page=snrPan, select=False, caption='Image SNR')
##camPanels.append((snrPan, 'SNR etc ...'))
##f.Show()
##time1.register_callback(snrPan.ccdPan.draw)
#""")

cm.join()
from PYME.Acquire.Hardware import lasers
scope.l488 = lasers.FakeLaser('488',scope.cam,1, initPower=10)
scope.l405 = lasers.FakeLaser('405',scope.cam,0, initPower=10)

scope.lasers = [scope.l405, scope.l488]

InitGUI("""
from PYME.Acquire.Hardware import LaserControlFrame
lcf = LaserControlFrame.LaserControlLight(MainFrame,scope.lasers)
time1.register_callback(lcf.refresh)
#lcf.Show()
camPanels.append((lcf, 'Laser Control'))
""")

InitGUI("""
lsf = lasersliders.LaserSliders(toolPanel, scope.lasers)
camPanels.append((lsf, 'Laser Powers'))
""")

InitGUI("""
from PYME.Acquire import sarcSpacing
ssp = sarcSpacing.SarcomereChecker(MainFrame, menuBar1, scope)
""")

InitGUI("""
from PYME.Acquire.Hardware import focusKeys
fk = focusKeys.FocusKeys(MainFrame, menuBar1, scope.piezos[0])
time1.register_callback(fk.refresh)
""")

InitGUI("""
from PYME.Acquire.Hardware import splitter
splt = splitter.Splitter(MainFrame, mControls, scope)
""")

InitGUI("""
from PYME.Acquire.Hardware import driftTracking, driftTrackGUI
scope.dt = driftTracking.correlator(scope, scope.fakePiezo)
dtp = driftTrackGUI.DriftTrackingControl(MainFrame, scope.dt)
camPanels.append((dtp, 'Focus Lock'))
""")



#scope.frameWrangler.WantFrameNotification.append(calcSum)

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#import numpy
#psf = numpy.load(r'd:\psf647.npy')
#psf = numpy.maximum(psf, 0.)
#from PYME.Analysis import MetaData
#fakeCam.rend_im.setModel(psf, MetaData.TIRFDefault)

time.sleep(.5)
scope.initDone = True


