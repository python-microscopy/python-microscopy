
from PYME.Acquire.protocol import *
from PYME.Acquire.Utils.pointScanner import CircularPointScanner
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Scanner(CircularPointScanner):
    def __init__(self, scan_radius_um=100):
        self.enabled_views = []
        self.scan_radius_um = scan_radius_um
    
    def set_state(self, views=(0), size=(256, 256), integration_time=0.004):
        # for now, force views, otherwise if we don't queue before second htsms-tile we'll be stuck on view 1
        self.enabled_views = [0, 1, 2, 3]  # scope.cam.active_views  
        self.roi_size = (256, 256)  # (scope.cam.size_x, scope.cam.size_y)
        self.integration_time = 0.0125  # scope.cam.GetIntegTime()

        logger.debug('CAMERA SETTINGS - views %s, size %s, integration time %f [s]' % (views, size, integration_time))
        scope.frameWrangler.stop()
        scope.cam.disable_multiview()
        scope.cam.enable_multiview(views)
        scope.cam.ChangeMultiviewROISize(size[0], size[1])
        scope.cam.SetIntegTime(integration_time)
        scope.frameWrangler.Prepare()
        scope.frameWrangler.start()

        fs = np.array((scope.cam.size_x, scope.cam.size_y))
        # calculate tile spacing such that there is ~30% overlap.
        tile_spacing = (1/np.sqrt(2)) * fs * np.array(scope.GetPixelSize())
        # take the pixel size to be the same or at least similar in both directions
        pixel_radius = int(self.scan_radius_um / tile_spacing.mean())
        logger.debug('Circular tiler target radius in units of (overlapped) FOVs: %d' % pixel_radius)

        CircularPointScanner.__init__(self, scope, pixel_radius, tile_spacing, 
                                            1, 0, False, True, trigger=True, 
                                            stop_on_complete=True, return_to_start=False)
        self.on_stop.connect(scope.spoolController.StopSpooling)
    
    def return_state(self):
        logger.debug('returning camera state')
        scope.frameWrangler.stop()
        scope.cam.disable_multiview()
        scope.cam.enable_multiview(self.enabled_views)
        scope.cam.ChangeMultiviewROISize(self.roi_size[0], self.roi_size[1])
        scope.cam.SetIntegTime(self.integration_time)
        # this is gross, but often setting 1.25 ms just hits 80 FPS first time
        scope.cam.SetIntegTime(self.integration_time)
        scope.frameWrangler.Prepare()
        scope.frameWrangler.start()



scanner = Scanner(scan_radius_um=500)

# T(frame, function, *args) creates a new task
taskList = [
    T(-1, scope.turnAllLasersOff),
    T(-1, scope.l405.SetPower, 1),
    T(-1, scanner.set_state, [1], (256, 256), 0.005),
    T(-1, scope.focus_lock.EnableLock),  # should already be enabled, but just in case
    T(-1, scope.l405.TurnOn),
    T(-1, scanner.genCoords),
    T(0, scanner.start),
    # T(maxint, scope.l642.SetPower, 600),
    T(maxint, scope.turnAllLasersOff),
    T(maxint, scanner.return_state)
]

#optional - metadata entries
metaData = [
    ('Protocol.DataStartsAt', 0)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, filename=__file__)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, 
                                        randomise=False, filename=__file__)
