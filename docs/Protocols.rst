.. _protocols:

Acquisition Protocols
*********************

Acquisition protocols are written in a simplified super-set of python and are used to control the behaviour of the 
microscope hardware during the course of an acquisiton. Protocols allow automated acquisitions by defining a set of
 *tasks* which are executed at given camera frame numbers. Each protocol lives in a separate python file such that 
arbitrary code can be imported to suit your needs for a given acquisition type.

Protocol task lists are a series of `PYME.Acquire.protocol.TaskListTask` objects, which are commonly accessed through
the alias `T`. Each `T` in the list of tasks takes the form `T(when, what, *args)` where 'when' is the camera frame
on which to execute the 'what', and 'what' can be given (several) parameters, '*args'. 

Timing
======
Protocol tasks can be executed at several times throughout an acquisition, including before the camera is restarted.

When a series is not being acquired, PYME runs the camera continuously by default. When a PYMEAcquire series starts spooling, 
the first thing that happens is the frame polling is stopped and the camera is stopped. 
At this point, any tasks in the protocol task list with a 'when' of `-1` are executed, allowing you to define the state of the
microscope before the acquisition begins.
After these `-1` tasks, PYME collects metadata from any components registered to provide 'start metadata' at the beginning of a series.
The time is noted, and the camera is started to begin the acquisition.

Throughout the acquisition, the number of camera frames that have been acquired so far is tracked, such that protocol tasks corresponding to
arbitrary frame numbers can be executed at the appropriate time.
Due to latency, the timing of acquisition tasks on specified frame numbers is somewhat approximate (especially for fast frame rates), however,
when an acquisition task is carried out, the computer time is noted. 
Using the time logged when the camera was restarted, this allows us to generate a timeline of when acquisition events were carried out.

Finally, when spooling is stopped, PYME stops collecting camera frames as a part of the series, but continues to execute protocol tasks with 
a frame number 'when' higher than was reached during the acquisition. Using an effectively infinite 'when' allows one to specify tasks which
should be carried out at the end of each acquisition, such as shuttering lasers.

Example
-------
An example protocol is shown in the code block below

.. code-block:: python
    from PYME.Acquire.protocol import *

    # T(when, what, *args) creates a new task. "when" is the frame number, "what" is a function to
    # be called, and *args are any additional arguments.
    taskList = [
        T(-1, scope.state.update, {
            'Lasers.MPB560.On': False,
            'Lasers.MPB560.Power': 550.0,
            'Lasers.MPB642.On': True,
            'Lasers.MPB642.Power': 575.0,
            'Multiview.ActiveViews': [0, 1, 2, 3],
            'Multiview.ROISize': [256, 256],
            'Camera.IntegrationTime': 0.00125,
        }),
        T(-1, scope.focus_lock.DisableLockAfterAcquiringIfEnabled),
        T(8000, scope.l560.TurnOn),
        T(maxint, scope.turnAllLasersOff),
        T(maxint, scope.focus_lock.EnableLock),
    ]

    metaData = [
        ('Protocol.DataStartsAt', 0),
    ]

    preflight = []  # no preflight checks

    # must be defined for protocol to be discovered
    PROTOCOL = TaskListProtocol(taskList, metaData, preflight, filename=__file__)
    PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 1, 1000, metaData, preflight, slice_order='triangle',
                                            require_camera_restart=False, filename=__file__)


In the above protocol, before camera frames are acquired, the 'MPB642' laser is turned on, the 'MPB560' laser is turned off, and several camera settings are enabled.
Then, a focus lock method is executed, which acquires the lock position (to assure the acquisition begins at a specified same axial position), and then unlocks.
After 8,000 camera frames, the 'MPB560' laser is turned on. When the acquisition ends, all lasers are turned off, and the focus lock is re-armed (so as to prepare for
automated movement to another region of the sample).

Preflight checks
================
Protocols allow specification of preflight checks before an acquisition is started. 
These conditionals are meant to be relatively simple to write, and are specified as a string which will be converted to code and executed.
For example:

.. code-block:: python
    #optional - pre-flight check
    #a list of checks which should be performed prior to launching the protocol
    #syntax: C(expression to evaluate (quoted, should have boolean return), message to display on failure),
    preflight = [
    C('scope.cam.GetEMGain() == scope.cam.DefaultEMGain', 'Was expecting an intial e.m. gain of %d' % scope.cam.DefaultEMGain),
    C('scope.cam.GetROIX1() > 1', 'Looks like no ROI has been set'),
    C('scope.cam.GetIntegTime() < .06', 'Camera integration time may be too long'),
    ]

If these checks are specified in a protocol, but the condition evaluates to False, (in the example above, say the camera integration time is longer than 60 ms),
then when a user clicks 'Start Spooling', a pop-up box is presented with an appropriate warning message to the user, at which point they may either
proceed with the acquisition anyway, or cancel it.
