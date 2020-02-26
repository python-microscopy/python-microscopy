import numpy as np


def flag_piezo_movement(data_source, events, fps=None):
    """
    Flags localizations detected on frames between ProtocolFocus and PiezoOnTarget events.

    Parameters
    ----------

    Returns
    -------

    """


    t = data_source['t']
    moving = np.zeros(len(t), dtype=bool)

    focus_frames, focus_positions, focus_times, ontarget_times, ontarget_positions = [], [], [], [], []
    for event in events:
        if str(event['EventName']) == 'ProtocolFocus':
            # ProtocolFocus description is 'frame#, position'
            f_frame, f_position = str(event['EventDescr']).split(',')
            focus_frames.append(float(f_frame))
            focus_positions.append(float(f_position))
            focus_times.append(event['Time'])

        if str(event['EventName']) == 'PiezoOnTarget':
            # PiezoOnTarget description is the position
            ontarget_positions.append(float(event['EventDescr']))
            ontarget_times.append(float(event['Time']))

    # convert to arrays
    focus_frames = np.asarray(focus_frames)
    focus_times = np.asarray(focus_times)
    focus_positions = np.asarray(focus_positions)
    ontarget_times = np.asarray(ontarget_times)
    ontarget_positions = np.asarray(ontarget_positions)

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
    focus_frames_s = np.sort(focus_frames)
    for f_ind in range(len(focus_frames_s)):
        # find the next on-target
        valid_ontargets = ontarget_frames > focus_frames_s[f_ind]
        ot_inds = np.argsort(focus_frames - focus_frames_s[f_ind])
        try:
            ot_ind = ot_inds[valid_ontargets][0]
            ontarget_frame = ontarget_frames[ot_ind]
        except IndexError:
            ontarget_frame = t.max() + 1  # flag to the last localization

        moving[focus_frames_s[f_ind]:ontarget_frame] = True

        focus[]

    return moving

def piezo_offset_correction(data_source, events):