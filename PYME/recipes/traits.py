""" Shim to give us a uniform way of importing traits (and replacing them if needed in the future)"""
from __future__ import absolute_import

try:
    from enthought.traits.api import *
except ImportError:
    from traits.api import *


class Input(CStr):
    pass

class Output(CStr):
    pass

class Unifile(CStr):
    pass