from PYME.Acquire.protocol import *
from PYME.Acquire.Utils.pointScanner import PointScanner
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Scanner(PointScanner):
    def __init__(self, steps=10):
        self._steps = steps
    
    def setup(self):
        fs = np.array((scope.cam.size_x, scope.cam.size_y))
        fov_size = 2 * np.mean(fs * np.array(scope.GetPixelSize()))

        PointScanner.__init__(self, scope, self._steps, 
                                      fov_size/self._steps, 1, 0, 
                                      False, True, trigger=True, 
                                      stop_on_complete=True, 
                                      return_to_start=True)
        self.on_stop.connect(scope.spoolController.StopSpooling)
        self.genCoords()

scanner = Scanner()

taskList = [
    T(-1, scope.state.update, {
        'Lasers.MPB560.On': True,
        'Lasers.MPB560.Power': 0.0,
        'Lasers.MPB642.On': True,
        'Lasers.MPB642.Power': 0.0,
        'Lasers.OBIS405.On': False,
        'Lasers.OBIS488.On': False,
        'Multiview.ActiveViews': [0, 1, 2, 3],
        'Multiview.ROISize': [256, 256],
        'Camera.IntegrationTime': 0.1,
    }),
    T(-1, scanner.setup),
    T(0, scanner.start),
    T(maxint, scope.turnAllLasersOff),
]

metaData = [
    ('Protocol.DataStartsAt', 0),
]

preflight = []  # no preflight checks

# must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight, filename=__file__)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 1, 3, metaData, preflight, slice_order='saw',
                                        require_camera_restart=True, filename=__file__)
