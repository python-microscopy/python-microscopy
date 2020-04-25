"""
A collection of hacks to make PYME work with data spooled from Labview on the Bewersdorf lab high throughput machine

Mostly copyright Andrew Barentine 2018
"""

import numpy as np
import six
import sys

def spoof_focus_from_metadata(mdh):
    """
    Only to be used if events cannot be found, as this is substantially less fool-proof than relying on ProtocolFocus
    events. Throws a RuntimeWarning if focus cannot be accurately spoofed.

    Parameters
    ----------
    mdh : PYME.IO.MetaDataHandler.MDHandlerBase

    Returns
    -------
    position : ndarray
        z positions in microns of each step
    frames : ndarray
        frame index at which each step begins

    """
    try:
        frames = np.arange(0, mdh['StackSettings.FramesPerStep'] * mdh['StackSettings.NumSteps'] * mdh[
            'StackSettings.NumCycles'], mdh['StackSettings.FramesPerStep'])
        position = None
        assert mdh['StackSettings.StepSize'] > 0  # np.arange will fail with a step size of 0 
        position = np.arange(mdh.getOrDefault('Protocol.PiezoStartPos', 0),
                             mdh.getOrDefault('Protocol.PiezoStartPos', 0) + mdh['StackSettings.NumSteps'] * mdh[
                                 'StackSettings.StepSize'], mdh['StackSettings.StepSize'])
        position = np.tile(position, mdh['StackSettings.NumCycles'])

        assert len(position) == len(frames)

    except (KeyError, AttributeError) as e:  # cast error as RuntimeWarning
        six.reraise(RuntimeWarning,  type(e)('Could not spoof focus position due to incomplete StackSettings metadata'),
                    sys.exc_info()[2])
    except AssertionError as e:  # cast error as RuntimeWarning and include more information
        six.reraise(RuntimeWarning,
                    type(e)('Error spoofing focus events -  frames: %s, positions %s' % (frames, position)),
                    sys.exc_info()[2])

    return position, frames