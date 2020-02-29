import numpy as np
from PYME.Analysis import piecewiseMapping as piecewise_mapping


def flag_piezo_movement(frames, events, metadata):
    """
    Flags localizations detected on frames between ProtocolFocus and PiezoOnTarget events.

    Parameters
    ----------
    frames: ndarray
        frame numbers, typically localization data_source['t']
    events: list or dict
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
    focus_times = np.asarray(focus_times)
    ontarget_times = np.asarray(ontarget_times)

    I_time_focus = np.argsort(focus_times)

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

        # flag frames, inclusively between firing the focus change and knowing we're settled
        moving[np.logical_and(frames >= frame, frames <= ontarget_frame)] = True

    return moving

def correct_target_positions(data_source, events, cycle_time, start_time):
    """
    ProtocolFocus event descriptions list the intended focus target. Some piezos have a target tolerance and log their
    landing position with PiezoOnTarget.

    Parameters
    ----------
    data_source: dict-like, PYME.IO.tabular
        container for localizations, must have 't' in units of frames.
    events: list or dict
        acquisition events

    Returns
    -------
    corrected_focus: ndarray
        focus positions for each localization in data_source.

    """
    try:
        corrected_focus = np.copy(data_source['focus'])
    except KeyError:
        corrected_focus = np.zeros(len(data_source['t']), dtype=float)

    ontarget_times, ontarget_positions = [], []
    for event in events:
        if event['EventName'] == 'PiezoOnTarget':
            ontarget_times.append(float(event['Time']))
            ontarget_positions.append(float(event['EventDescr']))

    # sort in time
    I = np.argsort(ontarget_times)
    ontarget_times = np.asarray(ontarget_times)[I]
    ontarget_positions = np.asarray(ontarget_positions)[I]

    # todo - use piecewise mapping for a standard time to frame conversion
    try:
        # We should always have a frame 0 ProtocolFocus
        start_time = focus_times[focus_frames == 0]
    except IndexError:
        # back-calculate in a pinch
        f0, t0 = focus_frames[I_time_focus][0], focus_times[I_time_focus][0]
        start_time = t0 - (f0 / fps)

    # convert to frames, and ceil so we flag edge-cases as still moving
    ontarget_frames = np.ceil((ontarget_times - start_time) * fps).astype(int)

    return focus
