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
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware
import scipy
import time

@init_hardware('Fake Piezos')
def pz(scope):
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    scope.fakePiezo = fakePiezo.FakePiezo(100)
    scope.register_piezo(scope.fakePiezo, 'z', needCamRestart=True)
    
    scope.fakeXPiezo = fakePiezo.FakePiezo(10)
    scope.register_piezo(scope.fakeXPiezo, 'x')
    
    scope.fakeYPiezo = fakePiezo.FakePiezo(10)
    scope.register_piezo(scope.fakeYPiezo, 'y')

pz.join() #piezo must be there before we start camera



@init_hardware('Fake Camera')
def cm(scope):
    import numpy as np
    from PYME.Acquire.Hardware.Simulator import fakeCam
    from PYME.Acquire.Hardware.Simulator import rend_im
    
    PIXELSIZE_NM = 75.
    
    rend_im.set_pixelsize_nm(PIXELSIZE_NM)
    scope.register_camera(fakeCam.FakeCamera(PIXELSIZE_NM*np.arange(-128.0, 768.0 + 128.0),
                                             PIXELSIZE_NM*np.arange(-128.0, 128.0),
                                             fakeCam.NoiseMaker(),
                                             scope.fakePiezo, xpiezo = scope.fakeXPiezo, ypiezo = scope.fakeYPiezo),'Fake Camera')

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


@init_gui('Simulation UI')
def sim_controls(MainFrame, scope):
    from PYME.Acquire.Hardware.Simulator import dSimControl
    dsc = dSimControl.dSimControl(MainFrame, scope)
    MainFrame.AddPage(page=dsc, select=False, caption='Simulation Settings')

@init_gui('Camera controls')
def cam_controls(MainFrame, scope):
    from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame
    scope.camControls['Fake Camera'] = AndorControlFrame.AndorPanel(MainFrame, scope.cam, scope)
    MainFrame.camPanels.append((scope.camControls['Fake Camera'], 'EMCCD Properties'))

@init_gui('Sample database')
def samp_db(MainFrame, scope):
    from PYME.Acquire import sampleInformation
    sampPan = sampleInformation.slidePanel(MainFrame)
    MainFrame.camPanels.append((sampPan, 'Current Slide'))

@init_gui('Analysis settings')
def anal_settings(MainFrame, scope):
    from PYME.Acquire.ui import AnalysisSettingsUI
    AnalysisSettingsUI.Plug(scope, MainFrame)

@init_gui('Fake DMD')
def fake_dmd(MainFrame, scope):
    from PYMEnf.Hardware import FakeDMD, DMDGui
    scope.LC = FakeDMD.FakeDMD(scope)
    
    LCGui = DMDGui.DMDPanel(MainFrame,scope.LC, scope)
    MainFrame.camPanels.append((LCGui, 'DMD Control', False))


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

@init_hardware('Lasers')
def lasers(scope):
    from PYME.Acquire.Hardware import lasers
    scope.l488 = lasers.FakeLaser('l488',scope.cam,1, initPower=10)
    scope.l488.register(scope)
    scope.l405 = lasers.FakeLaser('l405',scope.cam,0, initPower=10)
    scope.l405.register(scope)
    

@init_gui('Laser controls')
def laser_controls(MainFrame, scope):
    from PYME.Acquire.ui import lasersliders
    
    lcf = lasersliders.LaserToggles(MainFrame.toolPanel, scope.state)
    MainFrame.time1.register_callback(lcf.update)
    MainFrame.camPanels.append((lcf, 'Laser Control'))
    
    lsf = lasersliders.LaserSliders(MainFrame.toolPanel, scope.state)
    MainFrame.time1.register_callback(lsf.update)
    MainFrame.camPanels.append((lsf, 'Laser Powers'))

@init_gui('Focus Keys')
def focus_keys(MainFrame, scope):
    from PYME.Acquire.Hardware import focusKeys
    fk = focusKeys.FocusKeys(MainFrame, None, scope.piezos[0])


#InitGUI("""
#from PYME.Acquire.Hardware import splitter
#splt = splitter.Splitter(MainFrame, None, scope, scope.cam)
#""")

@init_gui('Action manager')
def action_manager(MainFrame, scope):
    from PYME.Acquire.ui import actionUI
    
    ap = actionUI.ActionPanel(MainFrame, scope.actions, scope)
    MainFrame.AddPage(ap, caption='Queued Actions')


#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#import numpy
#psf = numpy.load(r'd:\psf647.npy')
#psf = numpy.maximum(psf, 0.)
#from PYME.Analysis import MetaData
#fakeCam.rend_im.setModel(psf, MetaData.TIRFDefault)

#time.sleep(.5)
scope.initDone = True


