
from PYME.recipes import acquisition
from PYME.Acquire.ActionManager import ActionManager, ActionManagerServer

class fakescope(object):
    pass

def setup():
    scope = fakescope()
    am = ActionManager(scope)
    ams = ActionManagerServer(am, 9393, '127.0.0.1')
    return am, ams

def test_queue_acquisitions():
    from PYME.IO.tabular import DictSource
    from PYME.recipes.base import ModuleCollection
    import numpy as np
    import time

    try:
        action_manager, action_manager_server = setup()
        action_manager.paused = True

        d = DictSource({'x': np.arange(10), 'y': np.arange(10)})
        rec = ModuleCollection()
        rec.namespace['input'] = d

        rec.add_module(acquisition.QueueAcquisitions(rec))
        rec.save()
        time.sleep(1)
        task = action_manager.actionQueue.get_nowait()
    finally:
        action_manager_server.shutdown()
