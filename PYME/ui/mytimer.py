#!/usr/bin/python

##################
# mytimer.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import wx
import time



class MultiTimer(wx.Timer):
    """
    Timer which calls multiple handlers
    """
    def __init__(self, PROFILE=False):
        wx.Timer.__init__(self)
        self.WantNotification = []
        self.PROFILE=PROFILE
        self.times = {}

    def Notify(self):
        for a in self.WantNotification:
            ts = time.time()
            a()
            
            if self.PROFILE:
                te = time.time() - ts
                ar = repr(a)
                #if ar in times.keys():
                #    self.times[ar] = self.times[ar]+ te
                #else:
                self.times[ar] = te
                
    def register_callback(self, callback):
        self.WantNotification.append(callback)

mytimer=MultiTimer #alias for backwards compatibility

class SingleTargetTimer(wx.Timer):
    """
    Single shot timer which only calls a single function when it fires
    
    This is a very light-weight wrapper around wx.Timer to permit consistent usage with non GUI timers.
    
    """
    def __init__(self, target):
        wx.Timer.__init__(self)
        self._target = target
        
    def Notify(self):
        self._target()
        
    def start(self, delay_ms, single_shot=True):
        if single_shot:
            wx.CallAfter(self.Start, delay_ms, wx.TIMER_ONE_SHOT)
        else:
            wx.CallAfter(self.Start, delay_ms, wx.TIMER_CONTINUOUS)
        
    def stop(self):
        self.Stop()
        
def call_in_main_thread(callable, *args, **kwargs):
    """
    Alias to permit different frontends
    
    Parameters
    ----------
    callable
    args
    kwargs

    Returns
    -------

    """
    wx.CallAfter(callable, *args, **kwargs)