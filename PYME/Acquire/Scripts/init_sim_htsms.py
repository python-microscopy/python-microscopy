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
from PYME import config
import scipy
import time

# Set a microscope name which describes this hardware configuration (e.g. a room number or similar)
# Used with the splitting ratio database and in other places where a microscope identifier is required.
scope.microscope_name = 'PYMESimulator'

# set some defaults for PYMEAcquire
# uncomment the line below for high-thoughput style directory hashing
# config.config['acquire-spool_subdirectories'] = True

@init_hardware('Fake Piezos')
def pz(scope):
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    scope.fakePiezo = fakePiezo.FakePiezo(100)
    scope.register_piezo(scope.fakePiezo, 'z', needCamRestart=True)
    
    scope.fakeXPiezo = fakePiezo.FakePiezo(10000)
    scope.register_piezo(scope.fakeXPiezo, 'x')
    
    scope.fakeYPiezo = fakePiezo.FakePiezo(10000)
    scope.register_piezo(scope.fakeYPiezo, 'y')

pz.join() #piezo must be there before we start camera

@init_hardware('Fake Camera')
def cm(scope):
    import numpy as np
    from PYME.Acquire.Hardware.Simulator import fakeCam
    from PYME.Acquire.Hardware import multiview
    size = 256
    cam = fakeCam.FakeCamera(size, #70*np.arange(0.0, 4*256.0),
                                             size, #70*np.arange(0.0, 256.0),
                                             fakeCam.NoiseMaker(),
                                             scope.fakePiezo, xpiezo = scope.fakeXPiezo,
                                             ypiezo = scope.fakeYPiezo,
                                             pixel_size_nm=100.,
                                             illumFcn = 'ROIIllumFunction'
                                             )
    cam.SetEMGain(150)
    
    mv_cam = multiview.MultiviewWrapper(cam, multiview_info = {
                                                                'Multiview.NumROIs': 4,
                                                                'Multiview.ChannelColor': [0, 1, 1, 0],
                                                                'Multiview.DefaultROISize': (size, size),
                                                                'Multiview.ROISizeOptions': [128, 240, 256],
                                                                'Multiview.ROI0Origin': (0, 0),
                                                                'Multiview.ROI1Origin': (size, 0),
                                                                'Multiview.ROI2Origin': (2*size, 0),
                                                                'Multiview.ROI3Origin': (3*size, 0),
                                                            }, 
                                            default_roi= {
                                                            'xi' : 0,
                                                            'yi' : 0,
                                                            'xf' : size*4,
                                                            'yf' : size
                                                         })
    scope.register_camera(mv_cam,'Fake Camera')
    mv_cam.register_state_handlers(scope.state)

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


# @init_gui('Simulation UI')
# def sim_controls(MainFrame, scope):
#     from PYME.Acquire.Hardware.Simulator import dSimControl
#     dsc = dSimControl.dSimControl(MainFrame, scope)
#     MainFrame.AddPage(page=dsc, select=False, caption='Simulation Settings')
#
#     scope.dsc = dsc


@init_gui('Simulation UI')
def sim_controls(MainFrame, scope):
    from PYME.Acquire.Hardware.Simulator import simcontrol, simui_wx
    #pre-polulate for dSTORM using tweaked values
    #note, probabilities are [spontaneous/s, switching laser/Ws, readout laser/Ws]
    transition_tensor =  simcontrol.fluor.createSimpleTransitionMatrix(pPA=[1e9, 0, 0],
                        pOnDark=[0, 0, 0.1],
                        pDarkOn=[0.02,0.001, 0],
                        pOnBleach=[0, 0, 0.01])
    scope.simcontrol = simcontrol.SimController(scope, 
                                                transistion_tensor=transition_tensor,
                                                spectral_signatures=[[1, 0.05], [0.05, 1]],
                                                splitter_info=([0, 0, 500., 500.], [0, 1, 1, 0]))
    scope.simcontrol.change_num_channels(4)
    scope.simcontrol.set_psf_model(simcontrol.PSFSettings(zernike_modes={4:1.5}))
    dsc = simui_wx.dSimControl(MainFrame, scope.simcontrol, show_status=False)
    MainFrame.AddPage(page=dsc, select=False, caption='Simulation Settings')

    msc = simui_wx.MiniSimPanel(MainFrame, scope.simcontrol)
    MainFrame.camPanels.append((msc, 'Simulation'))

    from PYME.simulation import pointsets
    scope.simcontrol.point_gen = simcontrol.RandomDistribution(n_instances=25,region_size=70e3, 
                                                                generator=simcontrol.Group(generators=[pointsets.WiglyFibreSource(),
                                                                    simcontrol.AssignChannel(channel=1, generator=pointsets.SHNucleusSource())
                                                                    ]))
    scope.simcontrol.generate_fluorophores()
    
    scope.dsc = dsc

@init_gui('Camera controls')
def cam_controls(MainFrame, scope):
    from PYME.Acquire.Hardware.AndorIXon import AndorControlFrame
    scope.camControls['Fake Camera'] = AndorControlFrame.AndorPanel(MainFrame, scope.cam, scope)
    MainFrame.camPanels.append((scope.camControls['Fake Camera'], 'EMCCD Properties', False))

    MainFrame.AddMenuItem('Camera', 'Set Multiview', 
                          lambda e: scope.state.setItem('Multiview.ActiveViews', [0, 1, 2, 3]))
    MainFrame.AddMenuItem('Camera', 'Clear Multiview', 
                          lambda e: scope.state.setItem('Multiview.ActiveViews', []))


cm.join()

@init_gui('Multiview Selection')
def multiview_selection(MainFrame, scope):
    from PYME.Acquire.ui import multiview_select

    ms = multiview_select.MultiviewSelect(MainFrame.toolPanel, scope)
    MainFrame.time1.WantNotification.append(ms.update)
    MainFrame.camPanels.append((ms, 'Multiview Selection'))

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
    #MainFrame.time1.WantNotification.append(lcf.update)
    #MainFrame.camPanels.append((lcf, 'Laser Control'))
    
    lsf = lasersliders.LaserSliders(MainFrame.toolPanel, scope.state)
    MainFrame.time1.WantNotification.append(lsf.update)
    MainFrame.camPanels.append((lsf, 'Laser Control'))

@init_gui('Focus Keys')
def focus_keys(MainFrame, scope):
    from PYME.Acquire.Hardware import focusKeys
    fk = focusKeys.FocusKeys(MainFrame, scope.piezos[0])

@init_gui('Sample Metadata')
def sample_metadata(main_frame, scope):
    from PYME.Acquire.sampleInformation import SimpleSampleInfoPanel
    sampanel = SimpleSampleInfoPanel(main_frame)
    main_frame.camPanels.append((sampanel, 'Sample Metadata'))
    # Prefill the data for our simulated structure
    sampanel.slide.SetValue('HTSMS_Sim01')
    sampanel.notes.SetValue('Chan0: WiglyFibre, Chan1: SHNucleus')

@init_gui('Action manager')
def action_manager(MainFrame, scope):
    from PYME.Acquire.ui import actionUI
    
    ap = actionUI.ActionPanel(MainFrame, scope.actions, scope)
    MainFrame.AddPage(ap, caption='Queued Actions')


@init_gui('Tiling')
def action_manager(MainFrame, scope):
    from PYME.Acquire.ui import tile_panel
    
    ap = tile_panel.TilePanel(MainFrame, scope)
    MainFrame.aqPanels.append((ap, 'Tiling'))

@init_gui('Chained Analysis')
def chained_analysis(main_frame, scope):
    from PYME.Acquire.htsms.rule_ui import SMLMChainedAnalysisPanel, get_rule_tile, RuleChain
    from PYME.cluster.rules import RecipeRuleFactory, SpoolLocalLocalizationRuleFactory
    from PYME.IO.MetaDataHandler import DictMDHandler
    
    SMLMChainedAnalysisPanel.plug(main_frame, scope)


#must be here!!!
joinBGInit() #wait for anything which was being done in a separate thread


scope.initDone = True
