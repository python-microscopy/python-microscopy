
import numpy as np

EVENT_ORDER = ['EventDescr', 'EventName', 'Time']
EVENT_DTYPE = np.dtype([('EventDescr', 'U256'), ('EventName', 'U32'), ('Time', '<f8')])

def event_array_from_hdf5(hdf5):
    """

    :param hdf5: tables.file.File
        open hdf5 file
    :return: ndarray
        structured array of events
    """
    return event_array_from_table(hdf5.root.Events)

def event_array_from_table(table):
    """

    :param table: tables.table.Table
        Events table of an hdf5 file
    :return: ndarray
        structured array of events
    """
    events = np.empty(len(table), dtype=EVENT_DTYPE)
    for field in EVENT_ORDER:
        events[:][field] = table[:][field]
    return events

def event_array_from_list(event_list):
    """

    :param event_list: list
        Each element is an event in `EVENT_ORDER` order
    :return: ndarry
        structured array of events
    """
    events_array = np.empty(len(event_list), dtype=EVENT_DTYPE)
    for j, ev in event_list:
        events_array['EventName'][j], events_array['EventDescr'][j], events_array['Time'][j] = ev
    return events_array
