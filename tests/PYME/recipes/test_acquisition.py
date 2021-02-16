
from PYME.recipes import acquisition
from PYME.Acquire.ActionManager import ActionManager, ActionManagerServer

class fakescope(object):
    pass

action_manager = None
action_manager_server= None

def setup():
    global action_manager, action_manager_server
    scope = fakescope()
    action_manager = ActionManager(scope)
    action_manager_server = ActionManagerServer(action_manager, 9393, '127.0.0.1')
    #return am, ams
    
def teardown():
    action_manager_server.shutdown()
    

def test_queue_acquisitions():
    from PYME.IO.tabular import DictSource
    from PYME.recipes.base import ModuleCollection
    import numpy as np
    import time

    action_manager.paused = True

    d = DictSource({'x': np.arange(10), 'y': np.arange(10)})
    rec = ModuleCollection()
    rec.namespace['input'] = d

    rec.add_module(acquisition.QueueAcquisitions(rec))
    rec.save()
    time.sleep(1)
    task = action_manager.actionQueue.get_nowait()
    
    
