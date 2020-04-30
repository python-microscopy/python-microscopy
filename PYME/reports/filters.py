from PYME import config
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

DEFAULT_FILTERS = {}

# ---------- add some default filters
from PYME.Analysis import graphing_filters
import base64
from PYME.IO import rgb_image
def round_sf(num, sf=3):
    import math

    fmt = '%' + ('.%d' % sf) + 'g'
    # rnd = float(fmt % f)
    rnd = round(num, sf - int(math.floor(math.log10(num))) - 1)
    if rnd > 1e6:
        return fmt % rnd
    elif rnd >= 10 ** sf:
        return '%d' % rnd
    else:
        fmt = '%' + ('.%d' % (sf - math.floor(math.log10(rnd)) - 1)) + 'f'
        return fmt % rnd

DEFAULT_FILTERS['movieplot'] = graphing_filters.movieplot2
DEFAULT_FILTERS['plot'] = graphing_filters.plot
DEFAULT_FILTERS['b64encode'] = base64.b64encode
DEFAULT_FILTERS['base64_image'] = rgb_image.base64_image
DEFAULT_FILTERS['roundsf'] = round_sf

# ------------ load filters from plugins

def populate_filters():
    """
    Load default PYME filters as well as filters declared in the config tree.
    :return: dict
        keys are filter names, values are the filter handles
    """
    filters = DEFAULT_FILTERS.copy()
    
    for plugin, filts in config.get_plugin_report_filters():
        try:
            for modname, filter in filts:
                logger.debug('Trying to import %s' % modname)
                mod = import_module(modname)
            
                filtername = '.'.join([plugin, filter])
                if filtername in filters.keys():
                    raise RuntimeError('Filter %s already defined' % filtername)
                
                filters[filtername] = getattr(mod, filter)
        except:
            logger.error('Failed to load filters from plugin: %s' % plugin)
    return filters
