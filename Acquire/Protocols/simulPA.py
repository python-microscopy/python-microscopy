#import all the stuff to make this work
from PYME.Acquire.protocol import *

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(5, Ex, 'scope.cam.laserPowers[0] = 20'),
T(10, Ex, 'scope.cam.laserPowers[0] = 0')
]

#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList)