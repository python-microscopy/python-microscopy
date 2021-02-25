
from PYME.Acquire.protocol import *
from PYME.Acquire.Utils.pointScanner import CircularPointScanner
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Scanner(CircularPointScanner):
    def __init__(self, scan_radius_um=100):
        self.enabled_views = []
        self.scan_radius_um = scan_radius_um
    
    def setup(self):
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
        
        self.genCoords()

        self.check_focus_lock_ok()
    
    def check_focus_lock_ok(self):
        scope.focus_lock.EnableLock()  # make sure we have the lock on
        if not scope.focus_lock.LockOK():
            import time
            logger.debug('lock not OK, pausing for 10 s')
            time.sleep(10)
            logger.debug('starting reacquire sequence')
            scope.focus_lock.ReacquireLock()


scanner = Scanner(scan_radius_um=500)

# T(frame, function, *args) creates a new task
taskList = [
    T(-1, scope.turnAllLasersOff),
    T(-1, scope.state.update, {
        'Lasers.OBIS405.Power': 1.0,
        'Multiview.ActiveViews': [1],
        'Multiview.ROISize': [256, 256],
        'Camera.IntegrationTime': 0.005,
    }),
    T(-1, scope._stage_leveler.reacquire_focus_lock),
    T(-1, scanner.setup),
    T(-1, scope.l405.TurnOn),
    T(0, scanner.start),
    T(maxint, scope.turnAllLasersOff),
]

#optional - metadata entries
metaData = [
    ('Protocol.DataStartsAt', 0)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, filename=__file__)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, metaData, 
                                        randomise=False, filename=__file__)
