from PYME.Acquire.protocol import *

# T(when, what, *args) creates a new task. "when" is the frame number, "what" is a function to
# be called, and *args are any additional arguments.
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
    T(0, scope.focus_lock.DisableLock),
    T(maxint, scope.turnAllLasersOff),
    # T(maxint, scope.focus_lock.EnableLock),
    # T(maxint, scope.spoolController.LaunchAnalysis)
]

metaData = [
    ('Protocol.DataStartsAt', 0),
]

preflight = []  # no preflight checks

# must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight, filename=__file__)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 1, 3, metaData, preflight, slice_order='saw',
                                        require_camera_restart=True, filename=__file__)