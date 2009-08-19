#
#

# Get documentation string:
from info_pad import __doc__

# Import symbols from sub-module:
from pad import *

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)

