import threading
import numpy as np
import tables
import time

from PYME import config

# numpy/pytables event representation
EVENTS_DTYPE = np.dtype([('EventDescr', 'S256'), ('EventName', 'S32'), ('Time', '<f8')])

class SpoolEvent(tables.IsDescription):
    """Pytables description for Events table in spooled dataset"""
    EventName = tables.StringCol(32)
    Time = tables.Float64Col()
    EventDescr = tables.StringCol(256)



class EventLogger(object):
    """Event logging backend base class"""
    
    def __init__(self, time_fcn=time.time):
        """
        
        Parameters
        ----------
        time_fcn : function, default=time.time
            function used to generate event timestamps. Pass a different function to, e.g.
            spoof timestamps when simulating.
        """
        self._time_fcn = time_fcn
    
    def logEvent(self, eventName, eventDescr='', timestamp=None):
        """Log an event. Should be overriden in derived classes.

        .. note::

          In addition to the name and description, timing information is recorded for each event.

        Parameters
        ----------
        eventName : string
            short event name - < 32 chars and should be shared by events of the
            same type.
        eventDescr : string
            description of the event - additional, even specific information
            packaged as a string (<255 chars). This is commonly used to store
            parameters - e.g. z positions, and should be both human readable and
            easily parsed.

        """
        pass
    
    @classmethod
    def list_to_array(cls, event_list):
        events_array = np.empty(len(event_list), dtype=EVENTS_DTYPE)
    
        for j, ev in enumerate(event_list):
            events_array['EventName'][j], events_array['EventDescr'][j], events_array['Time'][j] = ev
    
        return events_array


class HDFEventLogger(EventLogger):
    """Event logging backend for hdf/pytables data storage

    Parameters
    ----------
    spool : instance of HDFSpooler.Spooler
        The spooler to ascociate this logger with

    hdf5File : pytables hdf file
        The open HDF5 file to write to
    """
    
    def __init__(self, spooler, hdf5File, time_fcn=time.time):
        """Create a new Events table.
  
  
        """
        EventLogger.__init__(self, time_fcn=time_fcn)
        self.spooler = spooler
        #self.scope = scope
        self.hdf5File = hdf5File
        self._event_lock = threading.Lock()
        
        self.evts = self.hdf5File.create_table(hdf5File.root, 'Events', SpoolEvent)
    
    def logEvent(self, eventName, eventDescr='', timestamp=None):
        """Log an event.

        Parameters
        ----------
        eventName : string
            short event name - < 32 chars and should be shared by events of the
            same type.
        eventDescr : string
            description of the event - additional, even specific information
            packaged as a string (<255 chars). This is commonly used to store
            parameters - e.g. z positions, and should be both human readable and
            easily parsed.


        In addition to the name and description, timing information is recorded
        for each event.
        """
        if eventName == 'StartAq':
            eventDescr = '%d' % self.spooler.imNum
        
        with self._event_lock:
            ev = self.evts.row
            
            ev['EventName'] = eventName
            ev['EventDescr'] = eventDescr
            
            if timestamp is None:
                ev['Time'] = self._time_fcn()
            else:
                ev['Time'] = timestamp
            
            ev.append()
            #print(ev, (eventName, eventDescr, timestamp))
            
            if config.get('HDF-ZealousFlush', False):
                # flush after every event - this is slow, but ensures that data is written to disk
                # and that all events which were logged prior to a crash are preserved.
                # It probably only makes sense to do this for very critical data (hence the default of False).
                self.evts.flush()
            

class MemoryEventLogger(EventLogger):
    """ Event backend which records events to memory, to be saved at a later time"""
    def __init__(self, spooler, time_fcn=time.time):#, scope):
        EventLogger.__init__(self, time_fcn=time_fcn)
        self.spooler = spooler
        #self.scope = scope
        
        self._events = []
        self._event_lock = threading.Lock()
    
    def logEvent(self, eventName, eventDescr='', timestamp=None):
        if eventName == 'StartAq' and eventDescr == '':
            eventDescr = '%d' % self.spooler.imNum
        
        if timestamp is None:
            timestamp = self._time_fcn()
        
        with self._event_lock:
            self._events.append((eventName, eventDescr, timestamp))

    @property
    def events(self):
        return self._events
    
    def to_JSON(self):
        import json
        return json.dumps(self._events)
    
    def to_recarray(self):
        return self.list_to_array(self._events)
    