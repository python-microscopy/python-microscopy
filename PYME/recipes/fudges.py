"""
This module contains fudges and workarounds - e.g. to deal with poorly formatted input data. They should be regarded as
temporary hacks until the problems in the upstream data that necessitated them can be fixed.

TLDR - If you need to use modules from this file you're doing it wrong.

"""

from .base import register_module, ModuleBase, Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, List, DictStrStr, DictStrList, ListFloat, ListStr, File

import numpy as np
from PYME.IO import tabular

@register_module('EstimateFrameZFromStepMetadata')
class EstimateFrameZFromStepMetadata(ModuleBase):
    """
    Estimate the z position of a frame from the stepping information in the metadata. Only really useful for HT data
    streamed from Labview with broken events & even then treat with caution.
    
    """
    inputImage = Input('input')
    outputName = Output('z_positions')
    
    def execute(self, namespace):
        from PYME.experimental import labview_spooling_hacks
        from PYME.Analysis import piecewiseMapping
        
        im = namespace[self.inputImage]
        
        position, frames = labview_spooling_hacks.spoof_focus_from_metadata(im.mdh)
        
        frame_nums = np.arange(im.data.shape[2])

        zm = piecewiseMapping.piecewiseMap(0, frames, position, im.mdh['Camera.CycleTime'], xIsSecs=False)
        
        z = zm[frame_nums]
        
        namespace[self.outputName] = tabular.ColumnSource(z=z)
        
@register_module('AddMissingEvents')
class AddMissingEvents(ModuleBase):
    """"
    Load events and add them to an input datasource. Event loading is normally handled during
    file loading from disk and this function is only present to allow filetypes which don't
    traditionally have events to load them.

    Parameters
    ----------
    input_name : PYME.IO.image.ImageStack or PYME.IO.tabular.TabularBase
    events : File
    output_name : PYME.IO.image.ImageStack or PYME.IO.tabular.TabularBase
    """
    input_name = Input('')
    events = File('')
    output_name = Output('with_events')

    def execute(self, namespace):
        from PYME.IO import events, unifiedIO
        from os import path

        # load events
        fp, ext = path.splitext(self.events_file)
        with unifiedIO.local_or_temp_filename(self.events_file) as f:
            if ext == '.json':
                import json
                new_events = events.event_array_from_list(json.load(self.events_file))
            elif ext == '.npy':
                new_events = events.as_array(np.load(self.events_file))
            elif ext == '.hdf' or ext == '.h5r' or ext == '.h5':
                from PYME.IO.h5rFile import H5RFile
                events.event_array_from_hdf5(H5RFile(self.events_file).events)

        if hasattr(namespace[self.input], 'events') and namespace[self.input].events != None:
            logger.debug('input already has events, concatenating new events with existing')
            # concatenate with existing events
            new_events = np.concatenate(events.as_array(namespace[self.input].events), new_events)
        
        namespace[self.output] = inp
        namespace[self.output].events = new_events  # TODO - are we OK with adding to og input or do we need to wrap with tabular/new imagestack?
