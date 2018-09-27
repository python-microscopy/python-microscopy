from PYME import config
if config.get('cluster-listing-no-countdir', False):
    raise RuntimeError('Not using countdir')

from countdir import *