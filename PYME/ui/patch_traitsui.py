"""
Monkey patch for traitsui.wx.constants, to fix a bug whereby text fields are not usable
in dark mode due to a hard-coded background color.

We do this by installing an import hook that will redirect imports of traitsui.wx.constants to
a local version which uses a more appropriate background color.
"""

import sys
import os.path

from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location

class MyMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if 'traitsui.wx.constants' in fullname:
            #print(fullname, path, target)

            return spec_from_file_location(fullname, os.path.join(os.path.dirname(__file__), 'traitsui_constants.py'))

        return None # we don't know how to import this
    
sys.meta_path.insert(0, MyMetaFinder())