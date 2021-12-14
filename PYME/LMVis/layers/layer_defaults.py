"""
Default configurations for new layers
=====================================

To over-ride the defaults, place a file called ``layer_defaults.yaml`` in your ``~\.PYME`` directory. This should be a .yaml
representation of a dictionary of default settings. An example file is shown below:

.. code-block:: yaml

    # the method/intent used to display all the localisations (default layer when opening PYMEVis)
    points: 
      method: spheres # change display mode to spheres
      size: 50 # change size to 50
      cmap: gist_rainbow # keep colourmap as gist_rainbow
      vertexColour: ['z', 't'] # change to using z coloring by default (rather than  ['t', 'z'])
      
    # the method/intent used to display points from a single colour channel
    points_channel: 
      method: pointsprites # change display method to pointsprites
      alpha: 0.2 # transparency
      size: 20 # size
      cmap: ['c', 'm', 'y', 'r', 'g', 'b'] # change order of colour selections (default = ['r', 'g', 'b', 'c', 'm', 'y'])


**NOTE:** you must currently specify **all** setings for a layer - not just the ones you want to change/override as we
we don't merge settings with the default settings.

TODO - do a merge of the dictionaries so that we can just specify over-rides

"""
from PYME.misc.colormaps import cm
from PYME import config
import yaml

_defaults = {
    'points' : {  # single channel (or unseparated channels), PYMEVis default layer
        'method' : 'points', # render as solid points
        'cmap' : 'gist_rainbow',
        'vertexColour' : ['t', 'z'],
    },
    'points_channel' : {
        'method' : 'points', # render as solid points
        'cmap' : list(cm.solid_cmaps),
    },
    'surface' :{
        'cmap' : list(cm.solid_cmaps),
    },
    'image' :{
        'cmap' : 'greys'
    },
    'image_channel' :{
        'cmap' : ['r', 'g', 'b', 'c', 'y', 'm'],
    },
}

defaults = config.load_config('layer_defaults.yaml', _defaults)

def new_layer_settings(intent, layer_index=0, ds_keys=[], overrides={}):
    """
    Get the settings to use for a new channel based on intent. 
    
    To allow customisation with fallbacks, intent is split with an underscore into <base intent>_<specialisation>. Calling code should try to specify
    a specialisation where appropriate, even if defaults for that specialisation do not exist.

    Parameters
    ==========

    intent : str
        the intent of this layer

    layer_index : int, optional
        the number of previously exisiting layers of this kind/intent (used for asigning sequential colourmaps to layers)

    ds_keys : list, optional
        the available data columns (usually data.keys()). Used to select which variable to colour points by.

    """
    try:
        settings = defaults[intent].copy()
    except KeyError:
        settings = defaults[intent.split('_')[0]].copy()
    
    if isinstance(settings.get('cmap', None), list):
        settings['cmap'] = settings['cmap'][layer_index%len(settings['cmap'])]

    if isinstance(settings.get('vertexColour', None), list):
        vcs = [vc for vc in settings['vertexColour'] if vc in ds_keys]
        print('vcs: ', vcs)
        if len(vcs) > 0:
            settings['vertexColour'] = vcs[0]
        else:
            settings['vertexColour'] = ''

    settings.update(overrides)
    
    return settings

