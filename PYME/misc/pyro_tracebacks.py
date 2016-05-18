#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################


import Pyro.util
import Pyro.constants
import sys
#import traceback

old_except_hook = sys.excepthook

def excepthook(ex_type, ex_value, ex_tb):
    """An exception hook you can set sys.excepthook to, to automatically print remote Pyro tracebacks
    Modified from the stock Pyro version to fall back on the normal tracebacks if this isn't a Pyro exception    
    """
    if not getattr(ex_value,Pyro.constants.TRACEBACK_ATTRIBUTE,None) is None:
        traceback="".join(Pyro.util.getPyroTraceback(ex_value,ex_type,ex_tb))
        sys.stderr.write(traceback)
    else:
        old_except_hook(ex_type, ex_value, ex_tb)
 
 
sys.excepthook = excepthook