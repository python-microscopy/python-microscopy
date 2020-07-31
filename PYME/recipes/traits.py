""" Shim to give us a uniform way of importing traits (and replacing them if needed in the future)"""
from __future__ import absolute_import
import numpy as np

try:
    from enthought.traits.api import *
except ImportError:
    from traits.api import *


class Input(CStr):
    pass

class Output(CStr):
    pass

class FileOrURI(File):
    '''Custom trait for files that can either be on disk or on the cluster
    
    Used for calibration data such as PSFs and shiftmaps which are static across multiple invocations of a recipe.
    
    Disables validation to work around traitsUI issues
    '''
    
    info_text = 'a local file name or pyme-cluster:// URI'
    
    def __init__(self, *args, **kwargs):
        kwargs['exists'] = True # file must exist
        File.__init__(self, *args, **kwargs)
    
    def validate(self, object, name, value):
        # Traitsui hangs up if a file doesn't validate correctly and doesn't allow selecting a replacement - disable validation for now :(
        # FIXME
        return value

class IntFloat(BaseFloat):
    default_value = 0.0
    info_text = 'Frankenstein to deal with automatic introspection'

    def set(self, obj, name, value):
        setattr(self, name, value)
        self.set_value(obj, name, value)

    def get(self, obj, name):
        val = self.get_value(obj, name)
        if val is None:
            val = self.default_value
        if val == np.round(val):
            val = int(val)
        return val