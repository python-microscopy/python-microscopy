"""Multi-consumer multi-producer dispatching mechanism

Forked from django.dispatch, which was originally based on pydispatch (BSD) https://pypi.org/project/PyDispatcher/2.0.1/
See license.txt for original license.

Maintained locally to avoid having to add the whole of django as a dependency, and the resulting potential dependency
resolution issues (as it stands, the python-microscopy ecosystem has two different django apps which depend on different
django versions, and we would need to pin to a very outdated django version for these - at this stage it is better to
leave django version wrangling as a conscious choice for the end user).
 
Modifified to remove dependencies on django.util and to import the separate `dispatch` module on py2 (as the django version is py >=3.4 only)
"""

import sys
if sys.version_info.major == 2:
    # use old dispatch library
    from dispatch import *
else:
    from .dispatcher import Signal, receiver  # NOQA
