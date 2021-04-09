import threading
import numpy as np
import tables


# numpy/pytables event representation
EVENTS_DTYPE = np.dtype([('EventDescr', 'S256'), ('EventName', 'S32'), ('Time', '<f8')])

class SpoolEvent(tables.IsDescription):
    """Pytables description for Events table in spooled dataset"""
    EventName = tables.StringCol(32)
    Time = tables.Time64Col()
    EventDescr = tables.StringCol(256)



class EventLogger(object):
    """Event logging backend base class"""
    
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
    
    def __init__(self, spool, hdf5File):
        """Create a new Events table.
  
  
        """
        self.spooler = spool
        #self.scope = scope
        self.hdf5File = hdf5File
        
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
        
        ev = self.evts.row
        
        ev['EventName'] = eventName
        ev['EventDescr'] = eventDescr
        
        if timestamp is None:
            ev['Time'] = sp.timeFcn()
        else:
            ev['Time'] = timestamp
        
        ev.append()
        self.evts.flush()

class MemoryEventLogger(EventLogger):
    """ Event backend which records events to memory, to be saved at a later time"""
    def __init__(self, spool):#, scope):
        self.spooler = spool
        #self.scope = scope
        
        self._events = []
        self._event_lock = threading.Lock()
    
    def logEvent(self, eventName, eventDescr='', timestamp=None):
        from PYME.Acquire import Spooler as sp
        
        if eventName == 'StartAq' and eventDescr == '':
            eventDescr = '%d' % self.spooler.imNum
        
        if timestamp is None:
            timestamp = sp.timeFcn()
        
        with self._event_lock:
            self._events.append((eventName, eventDescr, timestamp))
    
    def to_JSON(self):
        import json
        return json.dumps(self._events)
    
    def to_recarray(self):
        return self.list_to_array(self._events)