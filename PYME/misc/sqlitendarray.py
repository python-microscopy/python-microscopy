#!/usr/bin/python

###############
# sqlitendarray.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:35:57 2012

@author: dbad004
"""

import sqlite3
from numpy import ndarray
import zlib
from six.moves import cPickle as pickle
import six
import numpy as np

#teach sqlite about numpy arrays
def adapt_numarray(array):
    return sqlite3.Binary(zlib.compress(array.dumps()))

def convert_numarray(s):
    #print(type(s))
    try:
        #assume data is zipped
        uz = zlib.decompress(s)
        #print(uz)
        if six.PY2:
            return pickle.loads(uz)
        else:
            return pickle.loads(uz, encoding='bytes')
    except:
        #fall back and just try unpickling
        return pickle.loads(s)

sqlite3.register_adapter(ndarray, adapt_numarray)
sqlite3.register_converter("ndarray", convert_numarray)