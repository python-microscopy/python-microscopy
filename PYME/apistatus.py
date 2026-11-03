"""

This module defines a series of decorators used for documenting functions API status / maturity. The hope is that this
will provide a simple way of indicating which parts of PYME are safe to use from third party code (and which parts are
volatile).

"""


def _api_annotation(status_blurb):
    def dec(obj):
        """Decorator to mark the API status of a function/class
        
        Parameters
        ----------
        obj : object to be decorated
        
        """
        
        obj.__doc__ = obj.__doc__ + '\nAPI Status\n-----------\n' + status_blurb
        
        return obj
    
    return dec

api = _api_annotation('''
**API:** This is considered part of the external API and expected to be used 3rd party code. Any backwards incompatible
future changes will by clearly signalled with deprecation warnings for several releases prior to making breaking changes.
''')

internal = _api_annotation('''
**Internal:** This is not expected to be called from 3rd party code and may change without notice, use at you own risk.
''')

experimental = _api_annotation('''
**Experimental:** This is experimental functionality, or functionality under development. It might become part of the API
in the future, but for now expect some volatility. If you want to use this functionality please get in touch - you might
even be able to help shape the interface to best suit your usage.
''')

dirty_api = _api_annotation('''
**Dirty API:** This really should be API, but is also in need of refactoring as the current interface is gross, broken,
or inconsistent. Guarantees here are weaker than for the API class, but we will endeavour to ensure that a) equivalent
functionality is available after refactoring an b) breaking changes give rise to error messages that point to information
on the new usage.
''')

removal_candidate = _api_annotation('''
***Removal candidate:** This code is likely to disappear, potentially without warning. It has either been replaced with
 a better alternative, or is old, legacy code which is hard to justify maintaining.
''')