#import all the stuff to make this work
from PYME.Acquire.protocol import *
import numpy

#define a list of tasks, where T(when, what, *args) creates a new task
#when is the frame number, what is a function to be called, and *args are any
#additional arguments
taskList = [
T(-1, scope.l560.TurnOn),
T(maxint, scope.l560.TurnOff)
]


#must be defined for protocol to be discovered
PROTOCOL = TaskListProtocol(taskList)
PROTOCOL_STACK = ZStackTaskListProtocol(taskList, 20, 100, randomise = False)