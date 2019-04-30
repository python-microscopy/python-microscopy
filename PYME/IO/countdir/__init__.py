from PYME import config
if config.get('cluster-listing-no-countdir', False):
    raise ImportError('Not using countdir')

from .countdir import *