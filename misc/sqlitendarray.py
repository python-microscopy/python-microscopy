# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:35:57 2012

@author: dbad004
"""

import sqlite3
from numpy import ndarray
import zlib
import cPickle as pickle

#teach sqlite about numpy arrays
def adapt_numarray(array):
    return sqlite3.Binary(zlib.compress(array.dumps()))

def convert_numarray(s):
    try:
        #assume data is zipped
        return pickle.loads(zlib.decompress(s))
    except:
        #fall back and just try unpickling
        return pickle.loads(s)

sqlite3.register_adapter(ndarray, adapt_numarray)
sqlite3.register_converter("ndarray", convert_numarray)