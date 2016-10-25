

import logging

logger = logging.getLogger(__name__)




class eventFilter:
    """

    """
    def __init__(self, visFr):
        self.pipeline = visFr.pipeline


        logging.debug('Adding menu items for event filters')

        visFr.AddMenuItem('Extras', 'Identify transient frames', self.OnIDTransient,
                          helpText='Toss frames acquired during pifoc translation')


    def OnIDTransient(self, event=None):
        from PYME.experimental import eventFilterUtils
        eventFilterUtils.idTransientFrames(self.pipeline.selectedDataSource, self.pipeline.events,
                                           self.pipeline.mdh['StackSettings.FramesPerStep'])

def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.eventfilters = eventFilter(visFr)

