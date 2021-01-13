from PYME.Acquire.protocol import *

# T(when, what, *args) creates a new task. "when" is the frame number, "what" is a function to
# be called, and *args are any additional arguments.
taskList = [
    T(-1, scope.state.update, {
        'Lasers.MPB560.On': True,
        'Lasers.MPB560.Power': 550.0,
        'Lasers.MPB642.On': True,
        'Lasers.MPB642.Power': 575.0,
        'Multiview.ActiveViews': [0, 1, 2, 3],
        'Multiview.ROISize': [256, 256],
        'Camera.IntegrationTime': 0.00125,
    }),
    T(0, scope.focus_lock.DisableLock),
    T(maxint, scope.turnAllLasersOff),
    T(maxint, scope.focus_lock.EnableLock)
]

metaData = [
    ('Protocol.DataStartsAt', 0),
]

preflight = []  # no preflight checks

# must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData, preflight, filename=__file__)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 1, 800, metaData, preflight, slice_order='triangle',
                                        require_camera_restart=False, filename=__file__)
