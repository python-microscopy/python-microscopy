# -*- coding: utf-8 -*-
"""
Mock standins for traits classes for use in documentation generation
"""

class HasTraits(object):
    def __init__(self, *args, **kwargs):
        pass

HasStrictTraits = HasTraits
class CStr(object):
    def __init__(self, *args, **kwargs):
        pass

class Float(HasTraits):
    pass
class List(HasTraits):
    def __call__(self, *args, **kwargs):
        pass

Int = CStr
CInt = CStr
Bool = CStr
Str = CStr
Enum = CStr
DictStrFloat = CStr
DictStrStr = CStr
DictStrBool = CStr
DictStrList = CStr
DictStrAny = CStr
Instance = CStr
File = CStr
ListFloat = CStr
ListStr = CStr
ListInt = CStr
BaseFloat = CStr

Property = CStr
WeakRef = CStr
Dict = CStr

class BaseEnum(HasTraits):
    pass


from functools import wraps

def mock_decorator(*args, **kwargs):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function
    return decorator

on_trait_change = mock_decorator
observe = mock_decorator