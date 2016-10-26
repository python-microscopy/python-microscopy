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
    # need this next bit to work if events is a dictionary or a recarray:
    names = np.array(events['EventName'])
    descr = np.array(events['EventDescr'])
    focusChanges = descr[names == 'ProtocolFocus']

    # for ProtocolFocus events, description is 'frame#, position'
    t = np.copy(dataSource['t'])
    ti = np.ones_like(t, dtype=int)

    for fi in focusChanges:
        fi = float(fi.split(',')[0])  # only interested in frame# at the moment
        ti[np.logical_and(t >= fi, t < fi+fps)] -= 1

    dataSource.addColumn('isTransient', ti)
    return