#!/usr/bin/python

###############
# djangoRecarray.py
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



def qsToRecarray(qs):
    """converts a django query set into a record array"""

    fields = qs.model._meta.fields

    names = []
    formats = []

    for f in fields:
        fmt = np.array(f.to_python('1')).dtype.str
        if fmt == '|S1':
            fmt = '|S80'
        formats.append(fmt)
        names.append(f.name)

    #print formats
    dtype = np.dtype({'names':names, 'formats':formats})
    #print dtype
    
    data = np.zeros(len(qs), dtype)

    for i, r in enumerate(qs):
        for n in names:
            data[n][i] = r.__getattribute__(n)
        #l.append(np.array(vals, dtype))

    #return hstack(l)

    return data


