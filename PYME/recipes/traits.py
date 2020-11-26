""" Shim to give us a uniform way of importing traits (and replacing them if needed in the future)"""
from __future__ import absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    from enthought.traits.api import *
except ImportError:
    from traits.api import *


class Input(CStr):
    fast_validate = None
    
    def validate(self, object, name, value):
        value = CStr.validate(self, object, name, value)
        
        # make sure we're not assigning to an output of this module
        mod_outputs = getattr(object, 'outputs', [])
        if value in mod_outputs:
            # trying to assign input to output
            raise TraitError('Assigning "%s" to input "%s" would result in a circular reference (value is in module outputs).' % (value, name))
            
        # make sure we are not assigning to any downstream outputs
        recipe = getattr(object, '_parent', None)
        if recipe is not None:
            if value in recipe.downstream_outputs(list(object.outputs)):
                # trying to assign input to output
                raise TraitError('Assigning "%s" to input "%s" would result in a circular reference (value is in downstream outputs).' % (value, name))
            
        return value
    
        

class Output(CStr):
    fast_validate = None
    
    def validate(self, object, name, value):
        value = CStr.validate(self, object, name, value)
        
        # make sure we're not assigning to an input of this module
        mod_inputs = getattr(object, 'inputs', [])
        if value in mod_inputs:
            # trying to assign input to output
            raise TraitError(
                'Assigning "%s" to output "%s" would result in a circular reference (value is in module inputs).' % (value, name))
            
        
        # make sure we are not assigning to any downstream outputs
        recipe = getattr(object, '_parent', None)
        if recipe is not None:
            if value in recipe.upstream_inputs(list(object.inputs)):
                # trying to assign input to output
                raise TraitError(
                    'Assigning "%s" to output "%s" would result in a circular reference (value is in upstream inputs).' % (
                    value, name))
                
        
        return value

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

class _IntFloat(BaseFloat):
    # WARNING: This is a fudge used in skimage wrapping. It should not be used unless absolutely necessary and might dissappear if and when we 
    # find a better replacement. Has a 'private' (leading underscore) name to discourage use.
    # TODO - Can we inherit from Float instead to eliminate the `set()` method?
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
