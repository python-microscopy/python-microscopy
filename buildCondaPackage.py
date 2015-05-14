# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:39:19 2015

@author: david
"""

import os
import sys

sys.path.append('.')

from PYME import version

os.environ['PYME_VERSION'] = version.version

os.system('conda build .')