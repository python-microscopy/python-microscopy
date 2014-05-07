#!/usr/bin/python

###############
# parseAtdebug.py
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


def parseError(s):
    where, fcn, err = s.split('\t')
    
    filen = where.split('(')[0]
    
    return filen, where, fcn, err
    
def parseErrors(f):
    errs = {}
    i = 0
    
    s = f.readline()
    while not s == '':
        filen, where, fcn, err = parseError(s)

        errs[err] = (i, where, fcn)        
        
        i += 1 
        print(i)
        s = f.readline()
        
    return errs