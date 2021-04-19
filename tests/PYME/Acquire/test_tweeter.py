import logging
import time

import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FakeSpooler(object):
    def __init__(self):
        self.spooled = 0

    def StartSpooling(self):
        self.spooled += 1

class FakeScope(object):
    def __init__(self, spool_controller):
        self.spool_controller = spool_controller

sleep_time = 1

n_tasks = 10
condition = {
    'queue_condition': n_tasks,
    'queue_above': 1,
    'trigger_counts': 1,
    'trigger_above': -1,
    'action_filter': 'spoolController.StartSpooling',
    'message': 'Testing!'
}

@pytest.mark.skip
def test_tweet():
    from PYME.Acquire.htsms.tweeter import LazyScopeTweeter
    from PYME.Acquire.ActionManager import ActionManager

    actions = ActionManager(FakeScope(FakeSpooler()))
    tweeter = LazyScopeTweeter(actions.actionQueue, sleep_time=sleep_time)
    
    tweeter.add_tweet_condition(condition)
    time.sleep(sleep_time)
    assert len(tweeter.dorment_conditions) == 1 and len(tweeter.live_conditions) == 0
    for ti in range(n_tasks + condition['queue_above']):  # go over for queue_above
        actions.QueueAction('spoolController.StartSpooling', {})
    time.sleep(sleep_time)
    # check that queueing has been triggered
    assert len(tweeter.dorment_conditions) == 0 and len(tweeter.live_conditions) == 1
    # now start peeling off tasks
    for ti in range(tweeter.action_queue.qsize()):
        tweeter.action_queue.get_nowait()
    time.sleep(sleep_time)
    assert (len(tweeter.dorment_conditions)) == 0 and len(tweeter.live_conditions) == 0
