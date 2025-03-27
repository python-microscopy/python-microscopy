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


from PYME.Acquire.Hardware import fakeShutters
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware
import scipy
import time
import os
import sys

fakeShutters = fakeShutters


# @init_hardware('ThorlabsCamera')
# def thor_cam(scope):
    # from PYME.Acquire.Hardware import thorlabs_cam
    # cam = thorlabs_cam.ThorlabsCamera()
    # 
    # scope.register_camera(cam, 'sCMOS')
# 
# @init_hardware('Fake Piezos')
# def pz(scope):
    # from PYME.Acquire.Hardware.Simulator import fakePiezo
    # scope.fakePiezo = fakePiezo.FakePiezo(100)
    # scope.register_piezo(scope.fakePiezo, 'z', needCamRestart=True)
    # 
    # scope.fakeXPiezo = fakePiezo.FakePiezo(100)
    # scope.register_piezo(scope.fakeXPiezo, 'x')
    # 
    # scope.fakeYPiezo = fakePiezo.FakePiezo(100)
    # scope.register_piezo(scope.fakeYPiezo, 'y')
# """ """  """ """
# pz.join() #piezo must be there before we start camera
# """ """  """ """
# 
# @init_hardware('Fake Camera')
# def cm(scope):
    # import numpy as np
    # from PYME.Acquire.Hardware.Simulator import fakeCam
    # cam = fakeCam.FakeCamera(256, #70*np.arange(0.0, 4*256.0),
                                            #  256, #70*np.arange(0.0, 256.0),
                                            #  fakeCam.NoiseMaker(),
                                            #  scope.fakePiezo, xpiezo = scope.fakeXPiezo,
                                            #  ypiezo = scope.fakeYPiezo,
                                            #  pixel_size_nm=70.,
                                            #  )
    # cam.SetEMGain(150)
    # scope.register_camera(cam,'Fake Camera')
# 

# @init_gui('sCMOS Camera controls')
# def thor_cam_controls(MainFrame, scope):
#     import wx
#     # Generate an empty, dummy control panel
#     # TODO - adapt PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel or similar to allow options to be set.
#     # As it stands, we just use the default gain and readout settings.
#     scope.camControls['HamamatsuORCA'] = wx.Panel(MainFrame)
#     MainFrame.camPanels.append((scope.camControls['HamamatsuORCA'], 'ORCA Properties'))

@init_hardware('Camera')
def cam(scope):
    from PYME.Acquire.Hardware.uc480 import uCam480
    uCam480.init(cameratype='ueye')
    cam = uCam480.uc480Camera(0)
    scope.register_camera(cam, 'ueye')


@init_hardware('Lasers & Shutters')
def lasers(scope):
    from PYME.Acquire.Hardware import toptica_ibeam

    scope.l488 = toptica_ibeam.TopticaIBeamLaser('l488', portname='/dev/ttyUSB0', scopeState=scope.state)
    scope.l488.SetPower(0.001) #set initial power to 1mW
    scope.lasers = [scope.l488]

@init_gui('Laser controls')
def laser_controls(MainFrame, scope):
    from PYME.Acquire.ui import lasersliders
    
    # lcf = lasersliders.LaserToggles(MainFrame.toolPanel, scope.state)
    # MainFrame.time1.register_callback(lcf.update)
    # MainFrame.camPanels.append((lcf, 'Lasers', False, False))
    
    lsf = lasersliders.LaserSliders(MainFrame.toolPanel, scope.state)
    MainFrame.time1.register_callback(lsf.update)
    MainFrame.camPanels.append((lsf, 'Lasers', False, False))

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


time.sleep(.5)
scope.initDone = True
