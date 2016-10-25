import numpy as np

def idTransientFrames(dataSource, events, fps):
    """
    Adds a 'isTransient' column to the input datasource, so that localizations from frames which were acquired during
    pifoc translation can be selectively filtered
    Args:
        dataSource:
        mdh:
        events:

    Returns:
        nothing, but adds column to input datasource

    """
    # fps = mdh['StackSettings.FramesPerStep']  # frames per step
    focusChanges = events[events['EventName'] == 'ProtocolFocus']
    # for ProtocolFocus events, description is 'frame#, position'
    t = np.copy(dataSource['t'])
    ti = np.ones_like(t, dtype=int)
    for fi in focusChanges['EventDescr']:
        fi = float(fi.split(',')[0])  # only interested in frame# at the moment
        ti[np.logical_and(t >= fi, t < fi+fps)] -= 1

    dataSource.addColumn('isTransient', ti)
    return