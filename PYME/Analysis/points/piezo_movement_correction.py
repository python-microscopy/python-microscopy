import numpy as np
from PYME.Analysis import piecewiseMapping as piecewise_mapping


def flag_piezo_movement(frames, events, metadata):
    """
    Flags localizations detected on frames between ProtocolFocus and PiezoOnTarget events.

    Parameters
    ----------
    frames: ndarray
        frame numbers, typically localization data_source['t']
    events: list or structured ndarray
        acquisition events
    metadata: PYME.IO.MetaDataHandler
        metadata with 'Camera.CycleTime' and 'StartTime' entries

    Returns
    -------
    moving: ndarray
        boolean array where elements correspond to each element in `frames`. False indicates the piezo is stable
        while True flags elements in `frames` where the piezo doesn't have a well defined position.

    """

    moving = np.zeros_like(frames, dtype=bool)

    focus_frames, focus_times, ontarget_times, = [], [], []
    for event in events:
        # fixme - bytes don't belong here
        if event['EventName'] == b'ProtocolFocus':
            # ProtocolFocus description is 'frame#, position'
            f_frame, f_position = event['EventDescr'].split(b',')
            focus_frames.append(float(f_frame))
            focus_times.append(event['Time'])

        # fixme - bytes don't belong here
        if event['EventName'] == b'PiezoOnTarget':
            ontarget_times.append(float(event['Time']))

    # convert to arrays
    focus_frames = np.asarray(focus_frames, dtype=int)
    ontarget_times = np.asarray(ontarget_times)

    # convert on-target times to frames
    ontarget_frames = piecewise_mapping.times_to_frames(ontarget_times, events, metadata).astype(int)

    # now go through each ProtocolFocus and flag localizations in between that and the closest on-target
    for frame in focus_frames:
        # find the next on-target
        valid_ontargets = ontarget_frames > frame
        ot_inds = np.argsort(ontarget_frames - frame)
        try:
            ot_ind = ot_inds[valid_ontargets][0]
            ontarget_frame = ontarget_frames[ot_ind]
        except IndexError:
            ontarget_frame =frames.max() + 1  # flag to the last localization
        # fixme - do we need to do anything about StartAc events here?
        # flag frames, inclusively between firing the focus change and knowing we're settled
        moving[np.logical_and(frames >= frame, frames <= ontarget_frame)] = True

    return moving

def correct_target_positions(frames, events, metadata):
    """
    ProtocolFocus event descriptions list the intended focus target. Some piezos have a target tolerance and log their
    landing position with PiezoOnTarget.

    Parameters
    ----------
    data_source: dict-like, PYME.IO.tabular
        container for localizations, must have 't' in units of frames.
    events: list or structured ndarray
        acquisition events

    Returns
    -------
    corrected_focus: ndarray
        focus positions for each localization in data_source.

    """
    # fixme - remove all the bytes stuff
    # fixme - at the moment does not correct for offset piezo's offset between ProtocolFocus and OnTarget positions.
    #   since we normally reject frames before the OnTarget maybe not a huge deal, but might we worth fixing.
    ontarget_times, ontarget_positions = [], []
    for event in events:
        if event['EventName'] == b'PiezoOnTarget':
            ontarget_times.append(float(event['Time']))
            ontarget_positions.append(float(event['EventDescr']))

    ontarget_times = np.asarray(ontarget_times)
    ontarget_frames = piecewise_mapping.times_to_frames(ontarget_times, events, metadata).astype(int)

    # spoof ProtocolFocus events from PiezoOnTarget events
    bonus_events = np.empty(len(events) + len(ontarget_times),
                            dtype=[('EventName', 'S32'), ('Time', '<f8'), ('EventDescr', 'S256')])
    bonus_events[:len(events)][:] = events[:]
    bonus_events[len(events):]['EventName'] = b'ProtocolFocus'
    bonus_events[len(events):]['Time'] = ontarget_times
    bonus_events[len(events):]['EventDescr'] = [', '.join((str(f), str(p))) for f, p in zip(ontarget_frames, ontarget_positions)]


    focus_mapping = piecewise_mapping.GeneratePMFromEventList(bonus_events, metadata, metadata['StartTime'],
                                                              metadata.getOrDefault('Protocol.PiezoStartPos',
                                                                                    ontarget_positions[0]))

    return focus_mapping(frames)

def normalize_ontarget_events(events, metadata):
    ontarget_times, ontarget_positions = [], []
    for event in events:
        if event['EventName'] == b'PiezoOnTarget':
            ontarget_times.append(float(event['Time']))
            ontarget_positions.append(float(event['EventDescr']))

    offset_map = piecewise_mapping.GeneratePMFromEventList(events, metadata, 0, 0, eventName=b'PiezoOffsetUpdate',
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
