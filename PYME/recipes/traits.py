""" Shim to give us a uniform way of importing traits (and replacing them if needed in the future)"""

try:
    from enthought.traits.api import *
except ImportError:
    from traits.api import *


class Input(CStr):
    pass

class Output(CStr):
    pass