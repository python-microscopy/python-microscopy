import threading
from PYME.Acquire import eventLog
import time
import numpy as np

class StripTiler(object):
    def _fast_step(self, step):
        if step is None:
            return 0.2 #200nm default target pixel size on fast axis
        else:
            return step

    def _slow_step(self, step):
        if step is None:
            return 30 #default of 30 um strip widths - FIXME change this to use currently set ROI and pixel size
        else:
            return step

    def __init__(self, target_rect, scope, fast_axis='y', step_x=None, step_y=None, log_events=False):
        self.xmin, self.xmax, self.ymin, self.ymax = target_rect
        self.scope = scope
        self.log_events = log_events

        self.fast_axis = fast_axis

        if self.fast_axis == 'x':
            self.step_fast = self._fast_step(step_x)
            self.step_slow = self._slow_step(step_y)
            self.fast_min, self.fast_max = self.xmin, self.xmax
            self.slow_min, self.slow_max = self.ymin, self.ymax
            self.slow_axis='y'
        else:
            self.step_fast = self._fast_step(step_y)
            self.step_slow = self._slow_step(step_x)
            self.fast_min, self.fast_max = self.ymin, self.ymax
            self.slow_min, self.slow_max = self.xmin, self.xmax
            self.slow_axis='x'

        self.vel = (self.step_fast / scope.cam.GetCycleTime())  # target velocity in um/s

        self.slow_steps = np.arange(self.slow_min, self.slow_max, self.step_slow)

        self.i = 0

    def _thread_target(self):
        fast_pz, fast_chan, mult = self.scope.positioning[self.fast_axis]
        mult = abs(mult)
        old_vel = fast_pz.GetVelocity(fast_chan) #remember the old velocity
        fast_pz.SetVelocity(fast_chan, self.vel / abs(mult)) #set velocity to give desired pixel sampling

        slow_pz, slow_chan, smult = self.scope.positioning[self.slow_axis]
        smult = abs(smult)

        #do stripes, alternating direction
        for i, slow_v in enumerate(self.slow_steps):
            if (i % 2):
                fs, fe = self.fast_max, self.fast_min
            else:
                fe, fs = self.fast_max, self.fast_min

            #self.scope.pa.stop()
            print('Moving to %f, %f' % (fs, slow_v))
            fast_pz.MoveTo(fast_chan, fs/mult)
            slow_pz.MoveTo(slow_chan, slow_v/smult)

            # wait for the slow axis to get there
            print('waiting for stage [slow move]')
            while not slow_pz.onTarget:
                time.sleep(0.1)

            if self.log_events:
                #TODO - fixme for axis selection
                eventLog.logEvent('ScannerXPos', '%3.6f' % slow_v)
                eventLog.logEvent('ScannerYPos', '%3.6f' % fs)

            #self.scope.pa.start()

            print('Slow move done, making fast move')

            #start the move with acquisition running
            fast_pz.MoveTo(fast_chan, fe/mult)

            print('waiting for stage [fast move]')
            while not fast_pz.onTarget:
                #wait for the fast axis to get there
                time.sleep(.1)

        fast_pz.SetVelocity(fast_chan, old_vel) #restore old velocity
        print('Scan complete')

    def start(self):
        self._movement_thread = threading.Thread(target=self._thread_target)
        self._movement_thread.start()