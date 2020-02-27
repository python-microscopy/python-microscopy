import numpy as np


def flag_piezo_movement(data_source, events, fps=None):
    """
    Flags localizations detected on frames between ProtocolFocus and PiezoOnTarget events.

    Parameters
    ----------
    data_source: dict-like, PYME.IO.tabular
        container for localizations, must have 't' in units of frames.
    events: list or dict
        acquisition events
    Returns
    -------
    moving: ndarray
        boolean array where elements correspond to each localization in data_source. False indicates the piezo is stable
        while True flags localizations from frames where the piezo doesn't have a well defined position.

    """

    moving = np.zeros(len(data_source['t']), dtype=bool)

    focus_frames, focus_positions, focus_times, ontarget_times, ontarget_positions = [], [], [], [], []
    for event in events:
        if str(event['EventName']) == 'ProtocolFocus':
            # ProtocolFocus description is 'frame#, position'
            f_frame, f_position = str(event['EventDescr']).split(',')
            focus_frames.append(float(f_frame))
            focus_times.append(event['Time'])

        if str(event['EventName']) == 'PiezoOnTarget':
            ontarget_times.append(float(event['Time']))

    # convert to arrays
    focus_frames = np.asarray(focus_frames, dtype=int)
    focus_times = np.asarray(focus_times)
    ontarget_times = np.asarray(ontarget_times)

    I_time_focus = np.argsort(focus_times)

    # convert on-target times to frames
    if fps is None:
        # estimate FPS if it wasn't an input
        t0, t1 = focus_times[I_time_focus][:2]
        f0, f1 = focus_frames[I_time_focus][:2]
        fps = (f1 - f0) / (t1 - t0)

    try:
        # We should always have a frame 0 ProtocolFocus
        start_time = focus_times[focus_frames == 0]
    except IndexError:
        # back-calculate in a pinch
        f0, t0 = focus_frames[I_time_focus][0], focus_times[I_time_focus][0]
        start_time = t0 - (f0 / fps)

    # convert to frames, and ceil so we flag edge-cases as still moving
    ontarget_frames = np.ceil((ontarget_times - start_time) * fps).astype(int)

    # now go through each ProtocolFocus and flag localizations in between that and the closest on-target
    for frame in focus_frames:
        # find the next on-target
        valid_ontargets = ontarget_frames > frame
        ot_inds = np.argsort(ontarget_frames - frame)
        try:
            ot_ind = ot_inds[valid_ontargets][0]
            ontarget_frame = ontarget_frames[ot_ind]
        except IndexError:
            ontarget_frame = data_source['t'].max() + 1  # flag to the last localization

        # flag frames, inclusively between firing the focus change and knowing we're settled
        moving[np.logical_and(data_source['t'] >= frame, data_source['t'] <= ontarget_frame)] = True

    return moving

def correct_target_positions(data_source, events):
    """
    ProtocolFocus event descriptions list the intended focus target. Some piezos have a target tolerance and log their
    landing position with PiezoOnTarget.

    :param data_source:
    :param events:
    :return:
    """
    return
