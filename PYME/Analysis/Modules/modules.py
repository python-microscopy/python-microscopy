# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:40:34 2015

@author: david
"""

from . import base
from . import filters
from . import processing
from . import measurement
try:
    from . import skfilters
except ImportError:
    pass

from base import ModuleCollection

def getModuleNames():
    return base.all_modules.keys()