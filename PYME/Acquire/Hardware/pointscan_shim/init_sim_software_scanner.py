
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware
import time



@init_hardware('Pointscan Camera')
def pointscan_cam(scope):
    from PYME.Acquire.Hardware.pointscan_shim import pointscan_camera
    from PYME.Acquire.Hardware.pointscan_shim import software_scanner
    from PYME.Acquire.Hardware.Simulator import fakePiezo

    scope.fakeXPiezo = fakePiezo.FakePiezo(10000)
    scope.register_piezo(scope.fakeXPiezo, 'x')
    
    scope.fakeYPiezo = fakePiezo.FakePiezo(10000)
    scope.register_piezo(scope.fakeYPiezo, 'y')

    scope.fakeZPiezo = fakePiezo.FakePiezo(100)
    scope.register_piezo(scope.fakeYPiezo, 'z')

    sig_provider = software_scanner.TestSignalProvider(scope.fakeXPiezo, scope.fakeYPiezo)
    scanner = software_scanner.SoftwareStageScanner(scope.fakeXPiezo, scope.fakeYPiezo, sig_provider, kwargs={
        'scan_params': {
            'n_x': 11,  # [px]
            'n_y': 11
            }
        })
    cam = pointscan_camera.PointscanCameraShim(position_scanner=scanner)

    scope.register_camera(cam, 'Test')

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


time.sleep(.5)
scope.initDone = True
