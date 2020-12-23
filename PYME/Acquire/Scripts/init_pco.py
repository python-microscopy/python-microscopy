#!/usr/bin/python

from PYME.Acquire.ExecTools import joinBGInit, init_gui, init_hardware

from PYME import config

@init_hardware('Fake Piezos')
def pz(scope):
    from PYME.Acquire.Hardware.Simulator import fakePiezo
    scope.fakePiezo = fakePiezo.FakePiezo(100)
    scope.register_piezo(scope.fakePiezo, 'z', needCamRestart=True)
    
    scope.fakeXPiezo = fakePiezo.FakePiezo(100)
    scope.register_piezo(scope.fakeXPiezo, 'x')
    
    scope.fakeYPiezo = fakePiezo.FakePiezo(100)
    scope.register_piezo(scope.fakeYPiezo, 'y')

pz.join() #piezo must be there before we start camera

@init_hardware('PcoEdge42LT')
def pco_cam(scope):
    from PYME.Acquire.Hardware.pco.pco_edge_42_lt import PcoEdge42LT

    import logging
    logger = logging.getLogger(__name__)

    cam = PcoEdge42LT(0)
    cam.Init()

    # flip and rotate on primary camera should always be false - make the stage match the camera reference frame instead
    # as it's much easier
    # TODO - make flip, rotate etc actually work for tiling in case we have two cameras
    scope.register_camera(cam, 'PcoEdge42LT', rotate=False, flipx=False, flipy=False)

    logger.debug('here')

@init_gui('sCMOS Camera controls')
def pco_cam_controls(MainFrame, scope):
    import wx
    # Generate an empty, dummy control panel
    # TODO - adapt PYME.Acquire.Hardware.AndorNeo.ZylaControlPanel or similar to allow options to be set.
    # As it stands, we just use the default gain and readout settings.
    scope.camControls['PcoEdge42LT'] = wx.Panel(MainFrame)
    MainFrame.camPanels.append((scope.camControls['PcoEdge42LT'], 'pco.edge 4.2 LT Properties'))

joinBGInit() 

scope.initDone = True
