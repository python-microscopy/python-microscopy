# -*- coding: utf-8 -*-
"""
Code to spoof ctypes when generating documentation so that dlls do not need to
be present

Created on Wed May 25 17:29:08 2016

@author: david
"""
from ctypes import *
import ctypes

__version__ = ctypes.__version__

def doNothing(*args, **kwargs):
    pass

class CDLL(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattribute__(self, name):
        return doNothing
        
WinDLL = CDLL

def _dlopen(*args, **kwargs):
    return WinDLL()