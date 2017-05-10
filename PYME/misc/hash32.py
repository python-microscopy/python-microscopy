#!/usr/bin/python

###############
# hash32.py
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
import numpy as np
# noinspection PyCompatibility
from builtins import int

def c_mul(a, b):
    return eval(hex((int(a) * b) & 0xFFFFFFFF)[:-1])


def hashString32(s):
    if not s:
        return 0 # empty
    value = np.array([ord(s[0]) << 7], 'int32')
    for char in s:
        value = (value*1000003) ^ ord(char)
    value = value ^ len(s)
    value = int(value)
    if value == -1:
        value = -2
    return value