
import numpy as np

EVENT_ORDER = ['EventDescr', 'EventName', 'Time']
EVENT_DTYPE = np.dtype([('EventDescr', 'U256'), ('EventName', 'U32'), ('Time', '<f8')])

def event_array_from_hdf5(hdf5):
    """
    Create an events array from an hdf5 file containing an Events table node.

    Parameters
    ----------
    hdf5 : tables.file.File
        open hdf5 file

    Returns
    -------
    event_array : numpy.ndarry
        structured array of events
    """
    return event_array_from_table(hdf5.root.Events)

def event_array_from_table(table):
    """
    Create an events array from an hdf5 Events table

    Parameters
    ----------
    table : tables.table.Table
        Events table of an hdf5 file
    
    Returns
    -------
    event_array : numpy.ndarry
        structured array of events
    
    Notes
    -----
    Historically we have saved events in h5 and h5r files in different dtype orders
    so it is important that this load loop through to load in `EVENT_ORDER`.
    """
    events = np.empty(len(table), dtype=EVENT_DTYPE)
    for field in EVENT_ORDER:
        events[:][field] = table[:][field]
    return events

def event_array_from_list(event_list):
    """
    Create an events array from a list of events

    Parameters
    ----------
    event_list: list
        Each element is an event in `EVENT_ORDER` order
    
    Returns
    -------
    event_array: numpy.ndarry
        structured array of events
    """
    events_array = np.empty(len(event_list), dtype=EVENT_DTYPE)
    for j, ev in event_list:
        events_array['EventName'][j], events_array['EventDescr'][j], events_array['Time'][j] = ev
    return events_array

def as_array(events):
    """
    Get events as an `EVENT_DTYPE` numpy.ndarray

    Parameters
    ----------
    events : list, numpy.ndarray, tables.file.File, or tables.table.Table
        input events

    Returns
    -------
    event_array: numpy.ndarry
        structured array of events
    """
    # if we call np.asarray(events, dtype=EVENT_DTYPE) will we get things in the right order?
    if isinstance(events, np.ndarray):
        return events.astype(EVENT_DTYPE)
    if isinstance(events, list):
        return event_array_from_list(events)
    else:
        import tables
        if isinstance(events, tables.file.File):
            return event_array_from_hdf5(events)
        else:
            return event_array_from_table(events)
    raise TypeError('PYME.IO.events.asarray supports list, array, and open hdf5 files(/table nodes)')
