# -*- coding: utf-8 -*-
"""
Mock standins for traits classes for use in documentation generation
"""

class HasTraits(object):
    pass

class CStr(object):
    def __init__(self, *args, **kwargs):
        pass

Float = CStr
List = CStr
Int = CStr
Bool = CStr
Str = CStr
Enum = CStr
DictStrFloat = CStr
DictStrStr = CStr
DictStrBool = CStr
DictStrList = CStr
Instance = CStr
File = CStr
ListFloat = CStr
ListStr = CStr
ListInt = CStr


from functools import wraps

def mock_decorator(*args, **kwargs):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function
    return decorator

on_trait_change = mock_decorator