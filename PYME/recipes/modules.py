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
from . import surface_fitting
from . import acquisition
from . import pyramid
from . import pointcloud
try:
    from . import skfilters
except:
    logger.exception('Could not import skimage')
    pass

try:
    from . import machine_learning
except ImportError:
    pass

from .base import ModuleCollection

#load any custom recipe modules
for mn in config.get_plugins('recipes'):
    try:
        print('Trying to load 3rd party recipe module %s' % mn)
        m = __import__(mn, fromlist=mn.split('.')[:-1])
        print('Loaded 3rd party recipe module %s' % mn)
    except Exception as e:
        logger.exception('Error loading plugin: %s' % mn)

def getModuleNames():
    return base.all_modules.keys()
