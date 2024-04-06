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
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware

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
from PYME.IO import MetaDataHandler


#PIFoc
@init_hardware('PIFoc')
def pifoc(scope):
    import sys
    if sys.version_info.major > 2:
        from PYME.Acquire.Hardware.Piezos import offsetPiezoREST as offsetPiezo
    else:
        from PYME.Acquire.Hardware.Piezos import offsetPiezo
    scope.piFoc = offsetPiezo.getClient()
    scope.register_piezo(scope.piFoc, 'z')

pifoc.join()

@init_hardware('Fake Piezos')
def pz(scope):
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    scope.fakePiezo = scope.piFoc# fakePiezo.FakePiezo(100)
    #scope.register_piezo(scope.fakePiezo, 'z', needCamRestart=True)
    
    scope.fakeXPiezo = fakePiezo.FakePiezo(10000)
    scope.register_piezo(scope.fakeXPiezo, 'x')
    
    scope.fakeYPiezo = fakePiezo.FakePiezo(10000)
    scope.register_piezo(scope.fakeYPiezo, 'y')

pz.join() #piezo must be there before we start camera



@init_hardware('Cameras')
def cam(scope):
    from PYME.Acquire.Hardware.Simulator import fakeCam
    cam = fakeCam.FakeCamera(256, #70*np.arange(0.0, 4*256.0),
                                             256, #70*np.arange(0.0, 256.0),
                                             fakeCam.NoiseMaker(),
                                             scope.fakePiezo, xpiezo = scope.fakeXPiezo,
                                             ypiezo = scope.fakeYPiezo,
                                             pixel_size_nm=70.,
                                             )
    cam.SetEMGain(150)
    scope.register_camera(cam,'Fake Camera')

@init_gui('Simulation UI')
def sim_controls(MainFrame, scope):
    from PYME.Acquire.Hardware.Simulator import simcontrol, simui_wx
    # simulate some fluorescent fiducials
    #note, probabilities are [spontaneous/s, switching laser/Ws, readout laser/Ws]
    transition_tensor =  simcontrol.fluor.createSimpleTransitionMatrix(pPA=[1e9, 0, 0],
                        pOnDark=[0, 0,0],
                        pDarkOn=[0.02,0.001, 0],
                        pOnBleach=[0, 0, 0.00])
    scope.simcontrol = simcontrol.SimController(scope, 
                                                transistion_tensor=transition_tensor,
                                                spectral_signatures=[[1, 0.05], [0.05, 1]],
                                                splitter_info=([0, 0, 500., 500.], [0, 1, 1, 0]),
                                                excitation_crossections=(1, 200))
    #scope.simcontrol.change_num_channels(4)
    #scope.simcontrol.set_psf_model(simcontrol.PSFSettings(zernike_modes={4:1.5}))
    dsc = simui_wx.dSimControl(MainFrame, scope.simcontrol, show_status=False)
    MainFrame.AddPage(page=dsc, select=False, caption='Simulation Settings')

    msc = simui_wx.MiniSimPanel(MainFrame, scope.simcontrol)
    MainFrame.camPanels.append((msc, 'Simulation'))

    from PYME.simulation import pointsets
    scope.simcontrol.point_gen = simcontrol.Shift(dx=5000, dy=5000, generator=pointsets.RandomSource(numPoints=20))
    #scope.simcontrol.point_gen = simcontrol.RandomDistribution(n_instances=25,region_size=70e3, 
    #                                                            generator=simcontrol.Group(generators=[pointsets.WiglyFibreSource(),
                                                                    #simcontrol.AssignChannel(channel=1, generator=pointsets.SHNucleusSource())
    #                                                                ]))
    scope.simcontrol.generate_fluorophores()
    
    scope.dsc = dsc

@init_gui('Camera controls')
def cam_controls(MainFrame, scope):
    from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame
    scope.camControls['Fake Camera'] = AndorControlFrame.AndorPanel(MainFrame, scope.cam, scope)
    MainFrame.camPanels.append((scope.camControls['Fake Camera'], 'EMCCD Properties', False))

cam.join()


@init_hardware('Lasers')
def lasers(scope):
    from PYME.Acquire.Hardware import lasers
    scope.l642 = lasers.FakeLaser('l642',scope.cam,1, initPower=1)
    scope.l642.register(scope)
    #scope.l405 = lasers.FakeLaser('l405',scope.cam,0, initPower=10)
    #scope.l405.register(scope)
    

@init_gui('Laser controls')
def laser_controls(MainFrame, scope):
    from PYME.Acquire.ui import lasersliders
    
    #lcf = lasersliders.LaserToggles(MainFrame.toolPanel, scope.state)
    #MainFrame.time1.register_callback(lcf.update)
    #MainFrame.camPanels.append((lcf, 'Laser Control'))
    
    lsf = lasersliders.LaserSliders(MainFrame.toolPanel, scope.state)
    MainFrame.time1.register_callback(lsf.update)
    MainFrame.camPanels.append((lsf, 'Laser Control'))

@init_gui('Drift tracking')
def drift_tracking(MainFrame, scope):
    from PYME.Acquire.Hardware import driftTracking, driftTrackGUI
    scope.dt = driftTracking.Correlator(scope, scope.piFoc)
    dtp = driftTrackGUI.DriftTrackingControl(MainFrame, scope.dt)
    MainFrame.camPanels.append((dtp, 'Focus Lock'))
    MainFrame.time1.register_callback(dtp.refresh)

@init_gui('Focus Keys')
def focus_keys(MainFrame, scope):
    from PYME.Acquire.Hardware import focusKeys
    fk = focusKeys.FocusKeys(MainFrame, scope.piezos[0])



#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread

#scope.SetCamera('A')

time.sleep(.5)
scope.initDone = True
