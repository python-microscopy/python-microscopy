#import all the stuff to make this work
from PYME.Acquire.protocol import *
import numpy

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, scope.turnAllLasersOff),
T(0, SetEMGain(0)),
T(1, scope.l671.TurnOn),
T(60, scope.l671.TurnOff),
T(61, SetEMGain(150))
T(80, scope.l671.TurnOn)
T(90, MainFrame.pan_spool.OnBAnalyse, None)
]

#optional - metadata entries
metaData = [
('Protocol.DarkFrameRange', (61, 80)),
('Protocol.DataStartsAt', 91)
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList, metaData)
