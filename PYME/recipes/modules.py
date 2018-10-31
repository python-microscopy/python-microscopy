# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:40:34 2015

@author: david
"""

from PYME import config
import logging
logger = logging.getLogger(__name__)

from . import base
from . import filters
from . import processing
from . import measurement
from . import tracking
from . import tablefilters
from . import output
from . import localisations
from . import multiview
try:
    from . import skfilters
except ImportError:
    pass

try:
    from . import machine_learning
except ImportError:
    pass

from .base import ModuleCollection

#load any custom recipe modules
for mn in config.get_plugins('recipes'):
    try:
        m = __import__(mn, fromlist=mn.split('.')[:-1])
    except Exception as e:

        logger.exception('Error loading plugin: %s' % mn)

def getModuleNames():
    return base.all_modules.keys()
