#!/usr/bin/python

##################
# eventLog.py
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
"""
Log events from within PYMEAcquire.

To log an event:
 
 >>> from PYME.Acquire import eventLog
 >>> eventLog.logEvent("eventName", "eventDescr") #NOTE - leave timestamp blank unless you are doing something funky
 
To receive events:
 
>>> from PYME.Acquire import eventLog
>>> import time
>>>
>>> class MyReviever(object):
>>>     def __init__(self, ...):
>>>         #register as a receiver of events
>>>         eventLog.WantEventNofication.append(self)
>>>
>>>     def eventLog(self, eventName, eventDescr = '', timestamp=None):
>>>         # timestamp is usually not supplied, provide one
>>>         if timestamp is None:
>>>             timestamp = time.time() #this is the simplest thing you can do for a timestamp.
>>>
>>>         #your event handling logic here.
 
 
TODO - Make this a bit saner by providing either an `EventReceiver` base class, or a `register_event_handler()` function
"""

WantEventNotification = []

def register_event_handler(EventReceiver):
    """
    Register an event handler to receive events.
    
    Parameters
    ----------
    EventReceiver : object (PYME.IO.events.EventLogger or subclass thereof)
        an object with a `logEvent(eventName, eventDescr = '', timestamp=None)` method
    """
    WantEventNotification.append(EventReceiver)

def remove_event_handler(EventReceiver):
    """
    Remove an event handler from the list of event receivers.
    
    Parameters
    ----------
    EventReceiver : object (PYME.IO.events.EventLogger or subclass thereof)
        an object with a `logEvent(eventName, eventDescr = '', timestamp=None)` method
    """
    WantEventNotification.remove(EventReceiver)

def logEvent(eventName, eventDescr = '', timestamp=None):
    """
    Log an event to all event receivers.
    
    Parameters
    ----------
    eventName : str
        a name for the event
    eventDescr : str
        a description (optionally containing event data)
    timestamp : float / None
        a timestamp. Note that timestamps are usually handled in the spooler and the parameter should **not** be provided
        unless you want to do something funky and spoof a timestamp that doesn't match the actual time.

    Returns
    -------

    """
    for evl in WantEventNotification:
            evl.logEvent(eventName, eventDescr, timestamp = timestamp)