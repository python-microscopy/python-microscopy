from PYME import config
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

FILTERS = {}

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

FILTERS['movieplot'] = graphing_filters.movieplot2
FILTERS['plot'] = graphing_filters.plot
FILTERS['b64encode'] = base64.b64encode
FILTERS['base64_image'] = rgb_image.base64_image
FILTERS['roundsf'] = round_sf

# ------------ load filters from plugins

def get_filters():
    """
    Load default PYME filters as well as filters declared in the config tree.
    :return: dict
        keys are filter names, values are the filter handles
    """
    filters = FILTERS.copy()
    plugin_filter_info = config.get_report_filters()
    for m, filts in plugin_filter_info.items():
        try:
            logger.debug('Trying to import %s' % m)
            mod = import_module(m)
            for filter in filts:
                filters[filter] = getattr(mod, filter)
        except:
            logger.error('Failed to load filters from plugin: %s' % m)
    return filters
