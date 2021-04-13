
import tweepy
import yaml
import os
import time
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

class FakeAPI(object):
    def update_status(self, status):
        print('FakeTweet: %s' % status)


class LazyScopeTweeter(object):
    """
    Something to run in the background and send automated tweets based on actions queued. Could easily use dispatch to
    listen for the onQueueChanged signal from the action_manager, which would be worthwhile if we are adding queue items
    we want to track/tweet in real time, but this is such low priority that building in longer delays seems appropriate
    """
    def __init__(self, action_queue, credential_filepath=None, prepend_text='', sleep_time=60, safety=True):

        self.action_queue = action_queue
        self._prepend_text = prepend_text
        self.sleep_time = sleep_time

        # get our keys/tokens
        if credential_filepath is None:
            credential_filepath = os.path.join(os.path.expanduser("~"), '.PYME/twitter-credentials.yaml')


        try:
            with open(credential_filepath) as f:
                cred = yaml.safe_load(f)
            # authenticate
            auth = tweepy.OAuthHandler(cred['consumer_key'], cred['consumer_secret'])
            auth.set_access_token(cred['access_token'], cred['access_token_secret'])
            self._t_api = tweepy.API(auth)
        except IOError as e:
            if not safety:
                raise e
            logger.debug('scope tweeter running in debug mode - no tweets will be issues')
            self._t_api = FakeAPI()



        self.live_conditions = []
        self.dorment_conditions = []

        self._poll_thread = None
        self.polling = False
        self.safety = safety

        self.start_poll()

    def __del__(self):
        self._poll_thread.join()

    @property
    def actions(self):
        tasks = [task[1] for task in self.action_queue.queue]
        return np.unique(tasks, return_counts=True)

    def start_poll(self):
        self.polling = True
        self._poll_thread = threading.Thread(target=self._poll)
        self._poll_thread.start()

    def scope_tweet(self, text):
        if not self.safety:
            self._t_api.update_status(self._prepend_text + text)
        else:
            logger.debug('Safety on, tweet would be: %s' % (self._prepend_text + text))

    def add_tweet_condition(self, condition):
        """

        Parameters
        ----------
        condition: dict
            following keys:
                queue_condition: int
                    Number of tasks required to make this condition live, e.g. 10000 If we want to tweet after 10,000
                    cells have been imaged
                queue_above: int
                    1: queue when count is above queue_condition, 0: queue when count is equal to queue_condition,
                    -1: queue when count is below queue_condition
                trigger_counts: int
                    number of counts
                trigger_above: int
                    1: trigger when count is above trigger_counts, 0: trigger when count is equal to trigger_counts,
                    -1: trigger when count is below trigger_counts
                action_filter: str
                    Type of task to count, e.g. 'spoolController.start_spooling'. A null string, '', will count all
                    tasks.
                message: str
                    Message to tweet once triggered
                recurring: bool
                    Flag to recycle this condition after it is met and the tweet is sent
                    FIXME- not implemented yet

        Returns
        -------
        None

        """
        # check if the condition is live or not
        counts = self._get_count(condition['action_filter'])
        if self._check_live(counts, condition['queue_condition'], condition['queue_above']):
            self.live_conditions.append(condition)
        else:
            self.dorment_conditions.append(condition)

    def _get_count(self, action_filter='', actions=None, counts=None):
        """
        Return number of counts for a specific action/task or the total count of all tasks/actions
        Parameters
        ----------
        action_filter: str
            Optional filter to return counts only for a specific action

        Returns
        -------
        action_count: int
            number of tasks in the queue

        """
        if action_filter == '':
            return self.action_queue.qsize()

        if actions is None or counts is None:
            actions, counts = self.actions


        try:
            ci = np.argwhere(actions == action_filter)[:, 0][0]
            return counts[ci]
        except IndexError:
            # action is not in actions
            return 0



    def _check_live(self, counts, queue_condition, queue_above):
        diff = counts - queue_condition

        if queue_above == 0:
            if diff == 0:
                return True
            return False

        if diff > 0:
            if queue_above > 0:
                return True
            return False

        # counts is lower than condition, only queue if queue_above is negative
        if queue_above < 0:
            return True
        return False


    def _poll(self):
        while self.polling:
            self.update()
            time.sleep(self.sleep_time)

    def update(self):
        """
        Note that by being lazy, we can potentially miss some tweets, e.g. those which become "live" and then
        the condition is no longer met by the time we get to it here

        Returns
        -------

        """
        # if we don't have any conditions to check then we're already done.
        if len(self.live_conditions) == 0 and len(self.dorment_conditions) == 0:
            return

        print('n live: %d' % len(self.live_conditions))

        tasks, task_counts = self.actions
        # check if any of our conditions are "live"  # todo - prone to size changed during iteration?
        for ci, cond in enumerate(self.dorment_conditions):
            if cond['action_filter'] == '':
                pop = self._check_live(task_counts.sum(), cond['queue_condition'], cond['queue_above'])
            elif cond['action_filter'] not in tasks:
                pop =  self._check_live(0, cond['queue_condition'], cond['queue_above'])
            else:
                pop = self._check_live(self._get_count(cond['action_filter'], tasks, task_counts),
                                       cond['queue_condition'], cond['queue_above'])

            # shift task if it is now live
            if pop:
                print('queuing task!')
                print(self.actions)
                self.live_conditions.append(self.dorment_conditions.pop(ci))


        # check that none of our conditions have been met
        to_pop = []
        for ci, cond in enumerate(self.live_conditions):  # todo - prone to size changed during iteration?
            diff = self._get_count(cond['action_filter']) - cond['trigger_counts']
            if diff > 0 and cond['trigger_above'] > 0:
                to_pop.append(ci)
                self.scope_tweet(cond['message'])
            elif diff == 0 and cond['trigger_above'] == 0:
                to_pop.append(ci)
                self.scope_tweet(cond['message'])
            elif diff < 0 and cond['trigger_above'] < 0:
                to_pop.append(ci)
                self.scope_tweet(cond['message'])


        for ci in to_pop:
            self.live_conditions.pop(ci)



