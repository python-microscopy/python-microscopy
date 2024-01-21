.. _protocols:

Acquisition Protocols
*********************

Acquisition protocols are written in a simplified super-set of python and are used to control the behaviour of the 
microscope hardware during the course of an acquisiton. Protocols allow automated acquisitions by defining a set of
*tasks* which are executed at given camera frame numbers. 

Protocols are designed so that users can create a protocol without necessarily knowing how to code in python. That said,
more advanced users can include arbitrary python code.

Protocols are formatted as a list of tasks, each specified as `T(when, what, *args)` where 'when' is the camera frame
after which to execute the 'what', and 'what' can be given (several) parameters, `*args` (for the python-savy, `T` is an 
alias for creating a `PYME.Acquire.protocol.TaskListTask` object). 

Timing
======
Protocol tasks are executed when the corresponding frame (`when`) is retrieved from the camera by the software, and as a consequence
are not strictly synchronous with acquisition of the frame. There will typically be some latency due to buffering and 
the internal polling clock. In practice this latency is usually on the order of 50 - 100 ms or 1 frame (whichever is greater), but the 
only strong garuantees are that tasks will execute **after** acquisition of the corresponding frame, that tasks will be executed in their
frame order, and that all tasks will execute. 
To partially mitigate this latency, timestamps are recorded when each protocol task does execute, enabling 
us to generate a timeline of when aquisition events were carried out and to make an accurate post-hoc assignment between acquisition
events and and frame numbers. Should one wish to have tighter synchronisation during acqusition, cameras can be run in single-shot or 
software triggered modes which wait until preceeding protocol tasks have completed before triggering the next frame. This comes, however,
at the expense of significantly reduced overall acquisition speed.

There are 2 special case frame numbers for protocol tasks. Firstly, all tasks with a `when` of `-1` are executed prior to the start of the
acquisition, allowing you to define the state of the microscope before the acquisition begins. After these `-1` tasks, PYME collects metadata
from any components registered to provide 'start metadata' at the beginning of a series. The time is noted, and the camera is started to begin 
the acquisition. Secondly a value of `maxint` (technically any number higher than the acquisition length) can be used to specify tasks which
will be run after the end of each acquisition (for example, turning off lasers). This works because when spooling is stopped, PYME stops 
collecting camera frames, but continues to execute all remaining protocol tasks.

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
Protocols allow specification of preflight checks before an acquisition is started. The goal of the preflight checks is to allow sanity
checking on acquisition parameters to avoid common mistakes with parameter setting for a given acquisition type (protocol). If a preflight
check fails, the user will be prompted as to whether they want to continue with the acquisition or abort.
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

 
