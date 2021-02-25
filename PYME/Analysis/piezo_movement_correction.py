import numpy as np
from PYME.Analysis import piecewiseMapping as piecewise_mapping

# TODO - this module has a lot of small functions which do very little - are they needed?
# TODO - rename to something sensible and move (this is about event parsing, not analysis). Probably best refactored
# at some point in the future along with some of the other event handling stuff. Maybe to PYME.IO.events??

def flag_piezo_movement(frames, events, metadata):
    """
    Flags localizations detected on frames between ProtocolFocus and PiezoOnTarget events.

    Parameters
    ----------
    frames: ndarray
        frame numbers, typically localization data_source['t']
    events: list or structured ndarray
        acquisition events
    metadata: PYME.IO.MetaDataHandler.MDHandlerBase
        metadata with 'Camera.CycleTime' and 'StartTime' entries

    Returns
    -------
    moving: ndarray
        boolean array where elements correspond to each element in `frames`. False indicates the piezo is stable
        while True flags elements in `frames` where the piezo doesn't have a well defined position.

    """

    # FIXME - is this actually used anywhere?
    piezo_moving = map_piezo_moving(events, metadata)
    return piezo_moving(frames)

def map_piezo_moving(events, metadata):
    """
    Generates mapping function to flag whether input frames are between ProtocolFocus and PiezoOnTarget events.

    Parameters
    ----------
    events: list or structured ndarray
        acquisition events
    metadata: PYME.IO.MetaDataHandler.MDHandlerBase
        metadata with 'Camera.CycleTime' and 'StartTime' entries

    Returns
    -------
    piezo_moving: PYME.Analysis.piecewiseMapping.piecewiseMap
        callable object returning True for input frame numbers where the piezo is not settled.

    """
    # FIXME - is this actually used anywhere?
    return piecewise_mapping.bool_map_between_events(events, metadata, b'ProtocolFocus', b'PiezoOnTarget',
                                                     default=False)

def map_corrected_focus(events, metadata):
    """
    ProtocolFocus event descriptions list the intended focus target. Some piezos have a target tolerance and log their
    landing position with PiezoOnTarget. PiezoOffsetUpdate events are also accounted for.

    Parameters
    ----------
    events: list or structured ndarray
        acquisition events
    metadata: PYME.IO.MetaDataHandler.MDHandlerBase
        metadata with 'Camera.CycleTime' and 'StartTime' entries

    Returns
    -------
    focus_mapping: PYME.Analysis.piecewiseMapping.piecewiseMap
        callable function to return focus positions for each input frame number

    """
    # FIXME - this is only used below in correct_target_positions - is it needed?
    normalized_events = spoof_focus_events_from_ontarget(events, metadata)
    return piecewise_mapping.GeneratePMFromEventList(normalized_events, metadata, metadata['StartTime'],
                                                              metadata.getOrDefault('Protocol.PiezoStartPos', 0.))
def correct_target_positions(frames, events, metadata):
    """
    ProtocolFocus event descriptions list the intended focus target. Some piezos have a target tolerance and log their
    landing position with PiezoOnTarget. PiezoOffsetUpdate events are also accounted for.

    Parameters
    ----------
    frames: ndarray
        time array, e.g. data_source['t'], in units of frames
    events: list or structured ndarray
        acquisition events
    metadata: PYME.IO.MetaDataHandler.MDHandlerBase
        metadata with 'Camera.CycleTime' and 'StartTime' entries

    Returns
    -------
    corrected_focus: ndarray
        focus positions for each element in `frames`

    """
    focus_mapping = map_corrected_focus(events, metadata)
    return focus_mapping(frames)

def spoof_focus_events_from_ontarget(events, metadata):
    """
    Generates a acquisition event array where events from (offset) piezo's with on-target events are spoofed to
    look like standard ProtocolFocus events.

    Parameters
    ----------
    events: list or structured ndarray
        acquisition events
    metadata: PYME.IO.MetaDataHandler.MDHandlerBase
        metadata with 'Camera.CycleTime' and 'StartTime' entries
    Returns
    -------
    bonus_events: ndarray
        events with piezo offsets accounted for and PiezoOnTarget events spoofed as ProtocolFocus events

    Notes
    -----
    The on-target events are fired from standard piezo classes, not the OffsetPiezo subclasses, so the PiezoOnTarget
    positions and ProtocolFocus events have an offset between them which we remove in the output normalized events if
    there are PiezoOffsetUpdate events available to do so.

    """
    ontarget_times, ontarget_positions = [], []
    offset_start, first_offset_time = 0, np.finfo(float).max
    for event in events:
        if event['EventName'] == b'PiezoOnTarget':
            ontarget_times.append(float(event['Time']))
            ontarget_positions.append(float(event['EventDescr']))
        elif event['EventName'] == b'PiezoOffsetUpdate':
            if float(event['Time']) < first_offset_time:
                first_offset_time = float(event['Time'])
                offset_start = float(event['EventDescr'].decode('ascii').split(', ')[0])

    if len(ontarget_times) == 0:
        # nothing to do
        return events

    offset_map = piecewise_mapping.GeneratePMFromEventList(events, metadata, 0, offset_start, eventName=b'PiezoOffsetUpdate',
                                                           dataPos=0)

    ontarget_times = np.asarray(ontarget_times)
    ontarget_frames = piecewise_mapping.times_to_frames(ontarget_times, events, metadata).astype(int)


    # spoof ProtocolFocus events from PiezoOnTarget events
    bonus_events = np.empty(len(events) + len(ontarget_times),
                            dtype=[('EventName', 'S32'), ('Time', '<f8'), ('EventDescr', 'S256')])
    bonus_events[:len(events)][:] = events[:]
    bonus_events[len(events):]['EventName'] = b'ProtocolFocus'
    bonus_events[len(events):]['Time'] = ontarget_times
    bonus_events[len(events):]['EventDescr'] = [', '.join((str(f), str(p - offset_map(np.asarray([f]))[0]))) for f, p in
                                                zip(ontarget_frames, ontarget_positions)]

    return bonus_events
