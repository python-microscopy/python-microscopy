

import numpy as np
import logging

logger = logging.getLogger(__name__)

def idTransientFrames(dataSource, mdh, events):
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
    # fixme: are events always going to be sorted in time??
    fps = mdh['StackSettings.FramesPerStep']  # frames per step
    focusChanges = events[events['EventName'] == 'ProtocolFocus']
    # for ProtocolFocus events, description is 'frame#, position'
    t = np.copy(dataSource['t'])
    ti = np.ones_like(t, dtype=int)
    for fi in focusChanges['EventDescr']:
        fi = float(fi.split(',')[0])  # only interested in frame# at the moment
        ti[np.logical_and(t >= fi, t < fi+fps)] -= 1

    dataSource.addColumn('isTransient', ti)
    return


class eventFilter:
    """

    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline


        logging.debug('Adding menu items for event filters')

        visFr.AddMenuItem('Extras', 'Identify transient frames', self.OnIDTransient,
                          helpText='Toss frames acquired during pifoc translation')


    def OnIDTransient(self, event=None):

        idTransientFrames(self.pipeline.selectedDataSource, self.pipeline.mdh, self.pipeline.events)

def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.eventfilters = eventFilter(visFr)

